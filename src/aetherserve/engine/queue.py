"""准入队列：背压控制 + 加权优先级 + 老化防饿死。

设计要点
--------------------
**1. 为什么要准入控制而不是无限排队？**
GPU 吞吐是固定的。请求无限堆积的后果不是"变慢"，而是**延迟无界增长 + 内存耗尽**：
每个在途请求都要预留 KV Cache，队列越长显存压力越大，最终 OOM 把所有请求一起干掉。
所以选择在入口处**快速失败（fail fast）**：超过容量直接 429，让上游限流/降级/重试，
这是保护系统整体可用性的正确取舍（与 Hystrix/K8s 的 overload 保护同源）。

**2. 优先级怎么设计？**
堆内保存 ``(priority, seq, item)``，其中 ``priority`` 越小越优先。
有效优先级 = 基础优先级（租户等级、SLA 类别）− 老化奖励（等待时长 / aging_ms）。
等待越久奖励越大，因此低优先级请求的等待时间有上界，不会被饿死。

**3. 为什么用惰性堆而不是可删除堆？**
老化会让堆内元素的优先级持续变化，维护一个"可更新优先级的堆"需要索引堆，实现复杂。
这里采用惰性策略：弹出时用**当前时刻**重算优先级，只在"重算后更优先"时重新入堆。
由于 ``get()`` 每次只重塑队首，复杂度依然是 O(log n)，且无需额外索引结构。
"""

from __future__ import annotations

import asyncio
import heapq
import itertools
import time
from dataclasses import dataclass, field
from typing import Any

from ..gateway.schemas import ChatMessage
from ..observability import metrics
from .batching import BatchPlan, bucket_of

# 每次 get() 等待窗口的上限：保证"等满 max_wait_ms 就发车"能够被及时触发，
# 同时不引入忙等（窗口内没有新请求时会被定时唤醒重新判定）。
_MAX_WAIT_WINDOW_S = 0.05


class QueueFullError(RuntimeError):
    """队列或在途显存预算已满 —— 可预期的背压，网关应返回 429。"""

    def __init__(self, message: str, reason: str = "queue_full") -> None:
        super().__init__(message)
        self.reason = reason


@dataclass(slots=True)
class QueuedRequest:
    """队列中的一个待执行请求（调度元数据 + 载荷）。"""

    request_id: str
    model: str
    tenant: str
    messages: list[ChatMessage]
    params: Any
    prompt_tokens: int
    est_total_tokens: int
    priority: int = 0  # 越小越优先
    enqueued_at: float = field(default_factory=time.monotonic)
    start_event: asyncio.Event = field(default_factory=asyncio.Event)
    cancelled: bool = False

    @property
    def waited_ms(self) -> float:
        return (time.monotonic() - self.enqueued_at) * 1000.0

    @property
    def bucket(self) -> int:
        return bucket_of(self.prompt_tokens)


class AdmissionQueue:
    """带准入控制、加权优先级与老化的异步队列。"""

    def __init__(
        self,
        max_size: int = 512,
        max_tokens: int = 32768,
        aging_ms: float = 500.0,
    ) -> None:
        if max_size < 1 or max_tokens < 1:
            raise ValueError("max_size / max_tokens 必须为正数")

        self._heap: list[tuple[float, int, QueuedRequest]] = []
        self._seq = itertools.count()
        self._cond = asyncio.Condition()
        self._max_size = max_size
        self._max_tokens = max_tokens
        self._aging_ms = max(aging_ms, 1.0)
        self._inflight_tokens = 0
        self._closed = False

        # 统计字段（供 /metrics 与测试断言）
        self.accepted = 0
        self.rejected = 0

    # ------------------------------------------------------------ 只读属性
    @property
    def depth(self) -> int:
        return len(self._heap)

    @property
    def inflight_tokens(self) -> int:
        return self._inflight_tokens

    @property
    def capacity_left(self) -> int:
        return self._max_size - len(self._heap)

    @property
    def closed(self) -> bool:
        return self._closed

    # ------------------------------------------------------------ 内部工具
    def _effective_priority(self, item: QueuedRequest, now: float) -> float:
        """有效优先级 = 基础优先级 − 老化奖励。"""
        waited_ms = (now - item.enqueued_at) * 1000.0
        return item.priority - waited_ms / self._aging_ms

    def _sync_metrics(self) -> None:
        metrics.QUEUE_DEPTH.set(len(self._heap))
        metrics.INFLIGHT_TOKENS.set(self._inflight_tokens)

    def _discard(self, item: QueuedRequest) -> None:
        """从在途记账中移除一个请求（无论成功、失败还是取消）。"""
        if item.est_total_tokens:
            self._inflight_tokens = max(0, self._inflight_tokens - item.est_total_tokens)

    # ------------------------------------------------------------ 入队
    async def put(self, item: QueuedRequest) -> None:
        """准入并入队。

        Raises:
            QueueFullError: 队列长度或 KV Cache 预算超限（背压信号）。
        """
        async with self._cond:
            if self._closed:
                raise QueueFullError("队列已关闭", reason="closed")
            if len(self._heap) >= self._max_size:
                self.rejected += 1
                metrics.PREEMPTIONS_TOTAL.labels(reason="queue_full").inc()
                raise QueueFullError(f"准入队列已满（上限 {self._max_size}）", reason="queue_full")
            if self._inflight_tokens + item.est_total_tokens > self._max_tokens:
                self.rejected += 1
                metrics.PREEMPTIONS_TOTAL.labels(reason="kv_budget").inc()
                raise QueueFullError("KV Cache 预算不足", reason="kv_budget")

            self._inflight_tokens += item.est_total_tokens
            self.accepted += 1
            heapq.heappush(
                self._heap, (self._effective_priority(item, time.monotonic()), next(self._seq), item)
            )
            self._sync_metrics()
            self._cond.notify_all()

    # ------------------------------------------------------------ 出队
    async def get(self, plan: BatchPlan, timeout_s: float | None = None) -> list[QueuedRequest]:
        """阻塞取出一个批次。

        发车条件（满足任一）：
        - 批内请求数达到 ``max_batch_size``；
        - 批内 token 预算达到 ``max_batch_tokens``；
        - 队首请求等待时长超过 ``max_wait_ms``（保证低负载时不被无限等待）；
        - 队列已被取空（没有更多可聚合的请求）。

        Args:
            plan: 批构建约束。
            timeout_s: 整体等待上限；``None`` 表示一直等到有请求。

        Returns:
            已从队列摘除的请求列表，长度 ≥ 1。
        """
        deadline = None if timeout_s is None else time.monotonic() + timeout_s

        async with self._cond:
            while True:
                batch = self._drain(plan)
                if batch:
                    return batch
                if self._closed:
                    return []

                # 等待新请求或超时后重新评估。
                # 窗口取 min(max_wait_ms, 50ms)：既能在低延迟配置下及时发车，
                # 又避免在长等待配置下用极短超时造成无谓的自旋。
                wait_s = max(min(max(plan.max_wait_ms, 1.0) / 1000.0, _MAX_WAIT_WINDOW_S), 0.001)
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return []
                    wait_s = min(wait_s, remaining)
                try:
                    await asyncio.wait_for(self._cond.wait(), timeout=wait_s)
                except TimeoutError:
                    continue

    def _drain(self, plan: BatchPlan) -> list[QueuedRequest]:
        """在持锁状态下尝试摘取一个批次；不足则返回空列表。

        调用方必须已持有 ``self._cond``。

        实现采用"**惰性重排**"而非索引堆：弹出时用当前时刻重算有效优先级，
        若因老化而变得更优先则重新入堆并重试。由于每次只可能重塑队首，
        复杂度仍为 O(log n)，且无需维护"可更新优先级的堆"。
        """
        if not self._heap:
            return []

        now = time.monotonic()

        # ---- 惰性重排：把因老化而更优先的元素浮到堆顶 ----
        # 不变式：重排后 heap[0] 是全堆有效优先级最小的元素
        for _ in range(4):  # 有界循环：最多重塑 4 次，避免极端情况下自旋
            prio, seq, item = self._heap[0]
            if item.cancelled:
                heapq.heappop(self._heap)
                self._discard(item)
                if not self._heap:
                    self._sync_metrics()
                    return []
                continue
            new_prio = self._effective_priority(item, now)
            if new_prio < prio - 1e-9:
                heapq.heapreplace(self._heap, (new_prio, seq, item))
                continue
            break

        if not self._heap:
            self._sync_metrics()
            return []

        batch: list[QueuedRequest] = []
        deferred: list[tuple[float, int, QueuedRequest]] = []
        tokens = 0

        while self._heap and len(batch) < plan.max_batch_size:
            prio, seq, item = heapq.heappop(self._heap)

            if item.cancelled:
                self._discard(item)
                continue

            # 超预算则暂缓（首请求永远放行，避免大请求被永久阻塞）
            if batch and tokens + item.est_total_tokens > plan.max_batch_tokens:
                deferred.append((prio, seq, item))
                break

            batch.append(item)
            tokens += item.est_total_tokens

        for entry in deferred:
            heapq.heappush(self._heap, entry)

        if not batch:
            self._sync_metrics()
            return []

        # ---- 发车判定 ----
        # 批满 / token 预算满 / 队首等待超过 max_wait_ms（保证低负载下延迟有上界）
        #
        # 注意：这里**不能**加 "队列已空就发车" 的条件。
        # 队列短暂为空恰恰是"应该再等一会儿以聚合成批"的信号；
        # 若此时发车，低并发场景下每个请求都会单独成批，批处理收益归零。
        head_waited_ms = batch[0].waited_ms
        should_dispatch = (
            len(batch) >= plan.max_batch_size
            or tokens >= plan.max_batch_tokens
            or head_waited_ms >= plan.max_wait_ms
        )

        if not should_dispatch:
            # 未到发车时间：原样放回，等待新请求到达或等待窗口超时后重试
            for item in batch:
                heapq.heappush(
                    self._heap,
                    (self._effective_priority(item, now), next(self._seq), item),
                )
            self._sync_metrics()
            return []

        for item in batch:
            self._discard(item)
            metrics.QUEUE_WAIT.observe(item.waited_ms / 1000.0)
        self._sync_metrics()

        # 队首迟迟等不到后续请求时，get() 会因 wait 超时再次进入本函数，
        # 届时 waited_ms 必然超过 max_wait_ms，从而在此处正常发车。
        return batch

    # ------------------------------------------------------------ 生命周期
    async def wake(self) -> None:
        """唤醒正在等待的分发循环（例如有请求被取消后需要立即重新评估批次）。"""
        async with self._cond:
            self._cond.notify_all()

    async def cancel(self, request_id: str) -> bool:
        """标记取消（客户端断开 / 超时）。返回是否命中了在途请求。"""
        async with self._cond:
            for _, _, item in self._heap:
                if item.request_id == request_id:
                    item.cancelled = True
                    self._cond.notify_all()
                    return True
        return False

    async def close(self) -> None:
        """关闭队列并唤醒所有等待者（优雅停机）。"""
        async with self._cond:
            self._closed = True
            self._cond.notify_all()


__all__ = ["AdmissionQueue", "QueueFullError", "QueuedRequest"]
