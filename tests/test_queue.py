"""准入队列测试：容量、显存预算、优先级、老化、取消。"""

from __future__ import annotations

import asyncio

import pytest

from aetherserve.engine.batching import BatchPlan
from aetherserve.engine.queue import AdmissionQueue, QueuedRequest, QueueFullError


def make_item(
    request_id: str,
    tokens: int = 10,
    priority: int = 0,
    prompt_tokens: int | None = None,
) -> QueuedRequest:
    return QueuedRequest(
        request_id=request_id,
        model="echo-1",
        tenant="t1",
        messages=[],
        params=None,
        prompt_tokens=prompt_tokens if prompt_tokens is not None else tokens,
        est_total_tokens=tokens,
        priority=priority,
    )


PLAN = BatchPlan(max_batch_size=2, max_batch_tokens=1000, max_wait_ms=5.0, bucket_step=64)


async def wait_until(predicate, timeout_s: float = 2.0, interval_s: float = 0.001) -> bool:
    """轮询等待条件成立。

    避免依赖固定 sleep 时长：Windows 定时器与时钟粒度较粗（~15.6ms），
    基于"睡 N 毫秒后断言"的测试会间歇性失败。
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while loop.time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(interval_s)
    return predicate()


class TestAdmission:
    async def test_put_and_get(self, queue: AdmissionQueue) -> None:
        await queue.put(make_item("a"))
        batch = await queue.get(PLAN, timeout_s=1.0)
        assert [i.request_id for i in batch] == ["a"]
        assert queue.depth == 0
        assert queue.inflight_tokens == 0  # 出队后释放记账

    async def test_queue_full_backpressure(self) -> None:
        q = AdmissionQueue(max_size=2, max_tokens=10_000)
        await q.put(make_item("a"))
        await q.put(make_item("b"))
        with pytest.raises(QueueFullError) as ei:
            await q.put(make_item("c"))
        assert ei.value.reason == "queue_full"
        assert q.rejected == 1

    async def test_kv_budget_backpressure(self) -> None:
        """显存预算是硬约束：超过预算必须背压，而不是排队到 OOM。"""
        q = AdmissionQueue(max_size=100, max_tokens=100)
        await q.put(make_item("a", tokens=60))
        with pytest.raises(QueueFullError) as ei:
            await q.put(make_item("b", tokens=60))
        assert ei.value.reason == "kv_budget"
        assert q.inflight_tokens == 60  # 失败的请求不应污染记账

    async def test_inflight_tokens_tracked(self) -> None:
        q = AdmissionQueue(max_size=10, max_tokens=1000)
        await q.put(make_item("a", tokens=100))
        await q.put(make_item("b", tokens=200))
        assert q.inflight_tokens == 300


class TestBatchingBehaviour:
    async def test_waits_for_full_batch(self, queue: AdmissionQueue) -> None:
        """批未满时应等待更多请求，而不是立即发车（否则批处理没有收益）。"""
        # 用较长的 max_wait 放大"等待聚合"这一行为
        plan = BatchPlan(max_batch_size=2, max_batch_tokens=1000, max_wait_ms=500.0, bucket_step=64)
        await queue.put(make_item("a"))
        task = asyncio.create_task(queue.get(plan, timeout_s=2.0))

        # 在等待窗口内（远小于 500ms）任务不应完成
        await asyncio.sleep(0.05)
        assert not task.done(), "只有一个请求时不应立即发车"

        await queue.put(make_item("b"))  # 达到 max_batch_size=2
        batch = await asyncio.wait_for(task, timeout=2.0)
        assert len(batch) == 2

    async def test_dispatch_after_max_wait(self, queue: AdmissionQueue) -> None:
        """低负载下队首等待超过 max_wait_ms 必须发车，保证延迟上界。"""
        plan = BatchPlan(max_batch_size=8, max_batch_tokens=1000, max_wait_ms=60.0, bucket_step=64)
        await queue.put(make_item("solo"))
        t0 = asyncio.get_running_loop().time()
        batch = await queue.get(plan, timeout_s=3.0)
        elapsed_ms = (asyncio.get_running_loop().time() - t0) * 1000

        assert len(batch) == 1
        assert elapsed_ms >= 50.0, f"过早发车：{elapsed_ms:.1f}ms（应等待约 60ms）"
        assert elapsed_ms < 500.0, f"发车过晚：{elapsed_ms:.1f}ms"

    async def test_token_budget_splits_batch(self) -> None:
        plan = BatchPlan(max_batch_size=10, max_batch_tokens=100, max_wait_ms=5.0, bucket_step=64)
        q = AdmissionQueue(max_size=10, max_tokens=10_000)
        for i in range(4):
            await q.put(make_item(f"r{i}", tokens=60))
        batch = await q.get(plan, timeout_s=1.0)
        assert sum(i.est_total_tokens for i in batch) <= 120  # 至少一个但不超过太多

    async def test_empty_queue_get_with_timeout(self) -> None:
        q = AdmissionQueue(max_size=10, max_tokens=1000)
        batch = await q.get(PLAN, timeout_s=0.05)
        assert batch == []


class TestPriorityAndAging:
    async def test_priority_order(self) -> None:
        """基础优先级小的先出队。"""
        plan = BatchPlan(max_batch_size=10, max_batch_tokens=10_000, max_wait_ms=1.0, bucket_step=64)
        q = AdmissionQueue(max_size=10, max_tokens=10_000, aging_ms=10_000.0)
        await q.put(make_item("low", priority=10))
        await q.put(make_item("high", priority=1))
        await q.put(make_item("mid", priority=5))
        batch = await q.get(plan, timeout_s=1.0)
        assert [i.request_id for i in batch] == ["high", "mid", "low"]

    async def test_aging_prevents_starvation(self) -> None:
        """低优先级请求等待足够久后必须被优先处理（有界等待）。"""
        # max_wait_ms 设为 200ms：starved 等了 250ms，已超过发车阈值，
        # 因此 put(fresh) 后会立即发车，且顺序由有效优先级决定。
        plan = BatchPlan(max_batch_size=10, max_batch_tokens=10_000, max_wait_ms=200.0, bucket_step=64)
        q = AdmissionQueue(max_size=10, max_tokens=10_000, aging_ms=2.0)

        await q.put(make_item("starved", priority=100))  # 极低优先级
        # 老化奖励 = waited / aging_ms = 250 / 2 = 125 > 100，足以反超 fresh 的优先级 0
        await asyncio.sleep(0.25)
        await q.put(make_item("fresh", priority=0))  # 刚来的高优先级

        batch = await q.get(plan, timeout_s=1.0)
        assert [i.request_id for i in batch][:1] == ["starved"], "老化机制失效，低优先级请求被饿死"

    async def test_cancel_removes_inflight(self) -> None:
        q = AdmissionQueue(max_size=10, max_tokens=1000)
        await q.put(make_item("a"))
        assert await q.cancel("a") is True
        assert await q.cancel("missing") is False
        batch = await q.get(PLAN, timeout_s=0.2)
        assert all(i.request_id != "a" for i in batch)
        # 被取消的请求必须归还显存配额，否则队列会"假满"
        assert q.inflight_tokens == 0


class TestLifecycle:
    async def test_close_wakes_waiters(self) -> None:
        q = AdmissionQueue(max_size=10, max_tokens=1000)
        task = asyncio.create_task(q.get(PLAN, timeout_s=5.0))
        await asyncio.sleep(0.01)
        await q.close()
        assert await task == []

    async def test_put_after_close_rejected(self) -> None:
        q = AdmissionQueue(max_size=10, max_tokens=1000)
        await q.close()
        with pytest.raises(QueueFullError) as ei:
            await q.put(make_item("a"))
        assert ei.value.reason == "closed"

    async def test_invalid_construction(self) -> None:
        with pytest.raises(ValueError):
            AdmissionQueue(max_size=0)
        with pytest.raises(ValueError):
            AdmissionQueue(max_size=1, max_tokens=0)
