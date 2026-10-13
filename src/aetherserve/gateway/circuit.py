"""熔断器（Circuit Breaker）与重试策略。

为什么推理服务需要熔断？
------------------------
后端（vLLM 进程 / 远程推理服务）一旦变慢或挂掉，
**没有熔断的网关会把请求全部堆积在等待上**：连接池耗尽、
事件循环被 pending 请求占满、最终整个网关失去服务能力 —— 这就是级联故障。

熔断器的价值在于：当失败率超过阈值时**快速失败**，
把资源（连接、协程、显存配额）留给还有救的请求，并给后端恢复的时间窗。

状态机
------
    CLOSED ──(连续失败 ≥ threshold)──▶ OPEN
      ▲                                 │
      │                        (recovery_timeout 到期)
      │                                 ▼
      └──(half_open_max_calls 次探测全成功)── HALF_OPEN

半开态是"探针"：只放行少量请求，若成功则判定后端恢复，否则重新打开并延长冷却。
"""

from __future__ import annotations

import asyncio
import enum
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar

from ..observability import metrics
from ..observability.logging import get_logger

log = get_logger(__name__)

T = TypeVar("T")


class CircuitState(enum.IntEnum):
    """熔断状态（数值与 Prometheus 指标约定一致）。"""

    CLOSED = 0
    HALF_OPEN = 1
    OPEN = 2


class CircuitOpenError(Exception):
    """熔断打开，请求被快速拒绝（网关应返回 503）。"""

    def __init__(self, backend: str, retry_after_s: float = 1.0) -> None:
        super().__init__(f"后端 {backend} 熔断中，请稍后重试")
        self.backend = backend
        self.retry_after_s = retry_after_s


@dataclass
class CircuitStats:
    total_calls: int = 0
    total_failures: int = 0
    total_rejected: int = 0
    consecutive_failures: int = 0
    state_changes: int = 0


class CircuitBreaker:
    """单后端熔断器。"""

    def __init__(
        self,
        backend: str,
        failure_threshold: int = 5,
        recovery_timeout_s: float = 10.0,
        half_open_max_calls: int = 2,
    ) -> None:
        self.backend = backend
        self.failure_threshold = max(failure_threshold, 1)
        self.recovery_timeout_s = max(recovery_timeout_s, 0.1)
        self.half_open_max_calls = max(half_open_max_calls, 1)

        self._state = CircuitState.CLOSED
        self._opened_at = 0.0
        self._half_open_calls = 0
        self._lock = asyncio.Lock()
        self.stats = CircuitStats()
        self._publish_state()

    # ------------------------------------------------------------ 属性
    @property
    def state(self) -> CircuitState:
        return self._state

    @property
    def is_open(self) -> bool:
        return self._state is CircuitState.OPEN

    def _publish_state(self) -> None:
        metrics.CIRCUIT_STATE.labels(backend=self.backend).set(float(self._state))

    def _transition(self, new_state: CircuitState) -> None:
        if new_state is self._state:
            return
        log.warning(
            "circuit_state_change",
            backend=self.backend,
            old=self._state.name,
            new=new_state.name,
        )
        self._state = new_state
        self.stats.state_changes += 1
        if new_state is CircuitState.OPEN:
            self._opened_at = time.monotonic()
        elif new_state is CircuitState.HALF_OPEN:
            self._half_open_calls = 0
        elif new_state is CircuitState.CLOSED:
            self.stats.consecutive_failures = 0
        self._publish_state()

    # ------------------------------------------------------------ 准入
    async def acquire(self) -> None:
        """请求准入检查。

        Raises:
            CircuitOpenError: 当前处于打开状态（或半开态探测名额已满）。
        """
        async with self._lock:
            self.stats.total_calls += 1

            if self._state is CircuitState.OPEN:
                elapsed = time.monotonic() - self._opened_at
                if elapsed >= self.recovery_timeout_s:
                    self._transition(CircuitState.HALF_OPEN)
                else:
                    self.stats.total_rejected += 1
                    raise CircuitOpenError(self.backend, self.recovery_timeout_s - elapsed)

            if self._state is CircuitState.HALF_OPEN:
                if self._half_open_calls >= self.half_open_max_calls:
                    self.stats.total_rejected += 1
                    raise CircuitOpenError(self.backend, 0.5)
                self._half_open_calls += 1

    # ------------------------------------------------------------ 结果上报
    async def record_success(self) -> None:
        async with self._lock:
            self.stats.consecutive_failures = 0
            if self._state is CircuitState.HALF_OPEN:
                self._transition(CircuitState.CLOSED)

    async def record_failure(self) -> None:
        async with self._lock:
            self.stats.total_failures += 1
            self.stats.consecutive_failures += 1
            if self._state is CircuitState.HALF_OPEN:
                # 探测失败：立即重新打开，不给劣化后端机会
                self._transition(CircuitState.OPEN)
            elif self.stats.consecutive_failures >= self.failure_threshold:
                self._transition(CircuitState.OPEN)

    async def reset(self) -> None:
        """人工复位（运维接口 / 后端确认恢复后调用）。"""
        async with self._lock:
            self._transition(CircuitState.CLOSED)

    # ------------------------------------------------------------ 便捷用法
    async def call(self, fn: Callable[[], Awaitable[T]]) -> T:
        """包裹一次后端调用：自动准入、成功/失败上报。"""
        await self.acquire()
        try:
            result = await fn()
        except Exception:
            await self.record_failure()
            raise
        else:
            await self.record_success()
            return result

    def snapshot(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "state": self._state.name.lower(),
            "state_code": int(self._state),
            "consecutive_failures": self.stats.consecutive_failures,
            "total_calls": self.stats.total_calls,
            "total_failures": self.stats.total_failures,
            "total_rejected": self.stats.total_rejected,
            "failure_threshold": self.failure_threshold,
            "recovery_timeout_s": self.recovery_timeout_s,
        }


class CircuitBreakerRegistry:
    """按后端名称管理多个熔断器（多模型/多后端场景）。"""

    def __init__(self, failure_threshold: int = 5, recovery_timeout_s: float = 10.0) -> None:
        self._threshold = failure_threshold
        self._recovery = recovery_timeout_s
        self._breakers: dict[str, CircuitBreaker] = {}
        self._lock = asyncio.Lock()

    async def get(self, backend: str) -> CircuitBreaker:
        async with self._lock:
            cb = self._breakers.get(backend)
            if cb is None:
                cb = CircuitBreaker(
                    backend,
                    failure_threshold=self._threshold,
                    recovery_timeout_s=self._recovery,
                )
                self._breakers[backend] = cb
            return cb

    def snapshot(self) -> dict[str, object]:
        return {name: cb.snapshot() for name, cb in self._breakers.items()}
