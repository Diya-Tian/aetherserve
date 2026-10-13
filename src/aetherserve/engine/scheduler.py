"""调度器：批执行器的并发编排与显存水位决策。

职责边界
--------
- ``AdmissionQueue`` 负责"**谁先执行**"（公平性与背压）；
- ``Scheduler`` 负责"**同时执行几批**"（吞吐与显存的权衡）。

后者决定了工程师是否理解一个关键事实：
**并发批数不是越大越好。** 每增加一个并发批，KV Cache 峰值线性上升；
一旦超过显存上限，推理引擎会触发 swap/preemption，
反而导致尾延迟剧烈抖动（P99 恶化几十倍）。
因此调度器用"并发批数 × 单批 token 预算 ≈ 显存预算"的守恒关系来约束并发度，
并通过信号量 + 动态降级（水位过高时缩减批大小）实现软着陆。
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from ..observability import metrics
from ..observability.logging import get_logger
from .batching import BatchPlan

log = get_logger(__name__)


@dataclass(slots=True)
class SchedulerStats:
    """运行时统计快照，供 ``/v1/system/stats`` 与 Operator 伸缩决策使用。"""

    completed_batches: int = 0
    failed_batches: int = 0
    in_flight_batches: int = 0
    peak_in_flight_batches: int = 0
    peak_batch_size: int = 0
    total_requests: int = 0
    total_tokens: int = 0
    last_batch_tokens: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "completed_batches": self.completed_batches,
            "failed_batches": self.failed_batches,
            "in_flight_batches": self.in_flight_batches,
            "peak_in_flight_batches": self.peak_in_flight_batches,
            "peak_batch_size": self.peak_batch_size,
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
        }


class Scheduler:
    """批执行器的并发控制器。"""

    def __init__(
        self,
        plan: BatchPlan,
        concurrency: int = 4,
        kv_budget_tokens: int = 32768,
    ) -> None:
        if concurrency < 1:
            raise ValueError("concurrency 必须 >= 1")

        self.plan = plan
        self.concurrency = concurrency
        self.kv_budget_tokens = max(kv_budget_tokens, plan.max_batch_tokens)
        self._sem = asyncio.Semaphore(concurrency)
        self._stats = SchedulerStats()
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------ 属性
    @property
    def stats(self) -> SchedulerStats:
        return self._stats

    @property
    def available_slots(self) -> int:
        return self._sem._value

    # ------------------------------------------------------------ 自适应
    def effective_plan(self, inflight_tokens: int) -> BatchPlan:
        """按当前显存水位动态收缩批大小（软背压）。

        水位定义：``inflight_tokens / kv_budget_tokens``。
        - < 60%：使用完整批大小，追求吞吐；
        - 60%~85%：批大小线性衰减到一半，牺牲吞吐换稳定性；
        - > 85%：批大小降到最小（1~2），只保证长尾请求不被饿死。
        """
        if self.kv_budget_tokens <= 0:
            return self.plan

        ratio = inflight_tokens / self.kv_budget_tokens
        if ratio < 0.60:
            size = self.plan.max_batch_size
        elif ratio < 0.85:
            ratio_in_band = (ratio - 0.60) / 0.25
            size = max(2, int(self.plan.max_batch_size * (1.0 - 0.5 * ratio_in_band)))
        else:
            size = 1

        if size == self.plan.max_batch_size:
            return self.plan
        return BatchPlan(
            max_batch_size=size,
            max_batch_tokens=self.plan.max_batch_tokens,
            max_wait_ms=self.plan.max_wait_ms,
            bucket_step=self.plan.bucket_step,
        )

    # ------------------------------------------------------------ 记账
    async def on_batch_start(self, size: int, tokens: int) -> None:
        async with self._lock:
            s = self._stats
            s.in_flight_batches += 1
            s.total_requests += size
            s.total_tokens += tokens
            s.last_batch_tokens = tokens
            s.peak_in_flight_batches = max(s.peak_in_flight_batches, s.in_flight_batches)
            s.peak_batch_size = max(s.peak_batch_size, size)
        metrics.BATCH_SIZE.observe(size)
        metrics.BATCH_TOKENS.observe(tokens)

    async def on_batch_end(self, ok: bool) -> None:
        async with self._lock:
            s = self._stats
            s.in_flight_batches = max(0, s.in_flight_batches - 1)
            if ok:
                s.completed_batches += 1
            else:
                s.failed_batches += 1

    # ------------------------------------------------------------ 并发门
    async def acquire(self) -> float:
        """获取一个批执行槽位，返回排队等待时长（秒）。"""
        t0 = time.monotonic()
        await self._sem.acquire()
        return time.monotonic() - t0

    def release(self) -> None:
        self._sem.release()

    async def __aenter__(self) -> Scheduler:
        await self.acquire()
        return self

    async def __aexit__(self, *exc: object) -> None:
        self.release()

    def log_summary(self) -> None:
        log.info("scheduler_summary", **self._stats.as_dict(), concurrency=self.concurrency)


__all__ = ["Scheduler", "SchedulerStats"]
