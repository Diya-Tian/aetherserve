"""调度器测试：并发门限、自适应批大小（显存水位软背压）。"""

from __future__ import annotations

import asyncio

import pytest

from aetherserve.engine.batching import BatchPlan
from aetherserve.engine.scheduler import Scheduler


@pytest.fixture
def plan() -> BatchPlan:
    return BatchPlan(max_batch_size=10, max_batch_tokens=1000, max_wait_ms=10.0, bucket_step=64)


class TestConcurrencyGate:
    async def test_semaphore_limits_concurrency(self, plan: BatchPlan) -> None:
        sched = Scheduler(plan, concurrency=2)
        active = 0
        peak = 0

        async def worker() -> None:
            nonlocal active, peak
            async with sched:
                active += 1
                peak = max(peak, active)
                await asyncio.sleep(0.01)
                active -= 1

        await asyncio.gather(*(worker() for _ in range(6)))
        assert peak == 2

    async def test_acquire_returns_wait_time(self, plan: BatchPlan) -> None:
        """第二个请求必须等第一个释放后才能拿到槽位，等待时长应被记录。"""
        sched = Scheduler(plan, concurrency=1)
        await sched.acquire()

        async def second() -> float:
            return await sched.acquire()

        task = asyncio.create_task(second())
        await asyncio.sleep(0.03)
        assert not task.done(), "并发度为 1 时第二个请求不应立即拿到槽位"

        sched.release()
        waited = await task
        assert waited >= 0.02, f"等待时长未被正确记录: {waited}"
        sched.release()

    async def test_release_without_acquire_is_safe(self, plan: BatchPlan) -> None:
        sched = Scheduler(plan, concurrency=1)
        sched.release()  # 不应抛错，只是把信号量加 1

    async def test_invalid_concurrency(self, plan: BatchPlan) -> None:
        with pytest.raises(ValueError):
            Scheduler(plan, concurrency=0)


class TestAdaptiveBatchSize:
    """显存水位驱动的批大小收缩 —— 软背压的核心逻辑。"""

    def test_full_batch_at_low_watermark(self) -> None:
        plan = BatchPlan(max_batch_size=16, max_batch_tokens=8192, max_wait_ms=10.0)
        sched = Scheduler(plan, concurrency=4, kv_budget_tokens=100_000)
        assert sched.effective_plan(0).max_batch_size == 16
        assert sched.effective_plan(50_000).max_batch_size == 16  # 50% < 60%

    def test_shrinks_in_middle_band(self) -> None:
        plan = BatchPlan(max_batch_size=16, max_batch_tokens=8192, max_wait_ms=10.0)
        sched = Scheduler(plan, concurrency=4, kv_budget_tokens=100_000)
        mid = sched.effective_plan(70_000).max_batch_size  # 70% 落在 60%~85%
        assert 2 <= mid < 16, f"中间水位批大小应为 (2,16)，实际 {mid}"

    def test_minimal_at_high_watermark(self) -> None:
        plan = BatchPlan(max_batch_size=16, max_batch_tokens=8192, max_wait_ms=10.0)
        sched = Scheduler(plan, concurrency=4, kv_budget_tokens=100_000)
        assert sched.effective_plan(90_000).max_batch_size == 1
        assert sched.effective_plan(200_000).max_batch_size == 1

    def test_plan_preserves_other_fields(self) -> None:
        plan = BatchPlan(max_batch_size=16, max_batch_tokens=4096, max_wait_ms=25.0, bucket_step=64)
        sched = Scheduler(plan, concurrency=2, kv_budget_tokens=100_000)
        shrunk = sched.effective_plan(90_000)
        assert shrunk.max_batch_tokens == 4096
        assert shrunk.max_wait_ms == 25.0
        assert shrunk.bucket_step == 64

    def test_monotonic_shrink(self) -> None:
        """水位越高，批大小不应变大（单调性不变量）。"""
        plan = BatchPlan(max_batch_size=16, max_batch_tokens=8192, max_wait_ms=10.0)
        sched = Scheduler(plan, concurrency=4, kv_budget_tokens=100_000)
        sizes = [
            sched.effective_plan(int(r * 100_000)).max_batch_size for r in (0.1, 0.5, 0.7, 0.8, 0.9, 1.0)
        ]
        assert sizes == sorted(sizes, reverse=True), f"批大小非单调递减: {sizes}"


class TestStats:
    async def test_counts(self, plan: BatchPlan) -> None:
        sched = Scheduler(plan, concurrency=2)
        await sched.on_batch_start(4, 400)
        await sched.on_batch_start(2, 200)
        assert sched.stats.in_flight_batches == 2
        assert sched.stats.peak_in_flight_batches == 2
        assert sched.stats.peak_batch_size == 4
        assert sched.stats.total_requests == 6
        assert sched.stats.total_tokens == 600

        await sched.on_batch_end(ok=True)
        await sched.on_batch_end(ok=False)
        assert sched.stats.in_flight_batches == 0
        assert sched.stats.completed_batches == 1
        assert sched.stats.failed_batches == 1

    async def test_stats_never_negative(self, plan: BatchPlan) -> None:
        sched = Scheduler(plan, concurrency=2)
        await sched.on_batch_end(ok=True)
        assert sched.stats.in_flight_batches == 0
