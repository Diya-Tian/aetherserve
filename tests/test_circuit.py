"""熔断器状态机测试。"""

from __future__ import annotations

import asyncio
import contextlib

import pytest

from aetherserve.gateway.circuit import (
    CircuitBreaker,
    CircuitBreakerRegistry,
    CircuitOpenError,
    CircuitState,
)


async def wait_for_state(cb: CircuitBreaker, state: CircuitState, timeout_s: float = 2.0) -> None:
    """轮询等待状态迁移。

    为什么不用固定 ``asyncio.sleep``？
    Windows 的 ``time.monotonic`` 粒度约 15.6ms，
    固定 sleep 少量毫秒后判定时间是否"已过"会不稳定；轮询断言与时钟粒度解耦。
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while loop.time() < deadline:
        if cb.state is state or (await _peek_state(cb)) is state:
            return
        await asyncio.sleep(0.005)
    raise AssertionError(f"状态未在 {timeout_s}s 内变为 {state.name}，当前 {cb.state.name}")


async def _peek_state(cb: CircuitBreaker) -> CircuitState:
    """通过一次 acquire 触发惰性状态迁移（OPEN → HALF_OPEN）。"""
    # 熔断打开时 acquire 会抛异常，这正是我们要观测的"拒绝"行为，因此吞掉它
    with contextlib.suppress(CircuitOpenError):
        await cb.acquire()
    return cb.state


class TestCircuitBreaker:
    async def test_starts_closed(self) -> None:
        cb = CircuitBreaker("vllm")
        assert cb.state is CircuitState.CLOSED
        await cb.acquire()  # 不应抛错

    async def test_opens_after_threshold(self) -> None:
        cb = CircuitBreaker("vllm", failure_threshold=3)
        for _ in range(3):
            await cb.acquire()
            await cb.record_failure()
        assert cb.state is CircuitState.OPEN
        with pytest.raises(CircuitOpenError):
            await cb.acquire()
        assert cb.stats.total_rejected == 1

    async def test_success_resets_failure_counter(self) -> None:
        cb = CircuitBreaker("vllm", failure_threshold=3)
        await cb.record_failure()
        await cb.record_failure()
        await cb.record_success()
        await cb.record_failure()
        assert cb.state is CircuitState.CLOSED, "成功应清零连续失败计数"

    async def test_half_open_after_recovery_timeout(self) -> None:
        cb = CircuitBreaker("vllm", failure_threshold=1, recovery_timeout_s=0.05, half_open_max_calls=1)
        await cb.record_failure()
        assert cb.state is CircuitState.OPEN

        with pytest.raises(CircuitOpenError):
            await cb.acquire()

        await wait_for_state(cb, CircuitState.HALF_OPEN)

    async def test_half_open_success_closes(self) -> None:
        cb = CircuitBreaker("vllm", failure_threshold=1, recovery_timeout_s=0.02, half_open_max_calls=1)
        await cb.record_failure()
        await wait_for_state(cb, CircuitState.HALF_OPEN)
        await cb.record_success()
        assert cb.state is CircuitState.CLOSED

    async def test_half_open_failure_reopens_immediately(self) -> None:
        cb = CircuitBreaker("vllm", failure_threshold=2, recovery_timeout_s=0.02, half_open_max_calls=2)
        await cb.record_failure()
        await cb.record_failure()
        assert cb.state is CircuitState.OPEN

        await wait_for_state(cb, CircuitState.HALF_OPEN)
        await cb.record_failure()
        assert cb.state is CircuitState.OPEN, "半开探测失败必须立即重新打开"

    async def test_half_open_limits_probes(self) -> None:
        cb = CircuitBreaker("vllm", failure_threshold=1, recovery_timeout_s=0.02, half_open_max_calls=1)
        await cb.record_failure()
        await wait_for_state(cb, CircuitState.HALF_OPEN)
        with pytest.raises(CircuitOpenError):
            await cb.acquire()  # 探测名额已满

    async def test_call_wrapper_records_outcome(self) -> None:
        cb = CircuitBreaker("vllm", failure_threshold=2)

        async def ok() -> str:
            return "fine"

        assert await cb.call(ok) == "fine"
        assert cb.stats.total_failures == 0

        async def boom() -> None:
            raise RuntimeError("后端挂了")

        for _ in range(2):
            with pytest.raises(RuntimeError):
                await cb.call(boom)
        assert cb.state is CircuitState.OPEN

    async def test_reset(self) -> None:
        cb = CircuitBreaker("vllm", failure_threshold=1)
        await cb.record_failure()
        assert cb.state is CircuitState.OPEN
        await cb.reset()
        assert cb.state is CircuitState.CLOSED

    async def test_snapshot_shape(self) -> None:
        cb = CircuitBreaker("vllm", failure_threshold=2)
        await cb.record_failure()
        snap = cb.snapshot()
        assert snap["backend"] == "vllm"
        assert snap["state"] == "closed"
        assert snap["consecutive_failures"] == 1


class TestRegistry:
    async def test_creates_distinct_breakers(self) -> None:
        reg = CircuitBreakerRegistry(failure_threshold=1)
        a = await reg.get("vllm")
        b = await reg.get("transformers")
        a2 = await reg.get("vllm")
        assert a is a2
        assert a is not b

        await a.record_failure()
        assert a.state is CircuitState.OPEN
        assert b.state is CircuitState.CLOSED, "不同后端的熔断状态必须隔离"

    async def test_snapshot_all(self) -> None:
        reg = CircuitBreakerRegistry()
        await reg.get("vllm")
        await reg.get("echo")
        assert set(reg.snapshot()) == {"vllm", "echo"}
