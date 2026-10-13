"""连续批处理引擎 —— 项目核心组件。

一句话概括
----------
把"客户端到来的离散请求"聚合成"对推理后端的连续批次"，
同时逐个 token 地把结果**流式**回传给各自的调用方。

关键设计
--------
1. **两层解耦**：``AdmissionQueue`` 决定顺序（公平/背压），``Scheduler`` 决定并发（吞吐/显存）。
   引擎只做编排，不掺业务逻辑，因此每个部分都能独立压测与替换。

2. **每个请求一个私有 chunk 队列**：批执行器把不同请求的 token 分发到各自队列，
   调用方从自己的队列读取 —— 天然实现"批内多路复用"，
   慢客户端不会阻塞批内其他请求（对比：单一共享队列会互相干扰）。

3. **批内错误隔离**：``asyncio.gather(return_exceptions=True)``，
   单个请求失败只终结该请求，不影响同批其他请求，也不会让批执行器任务泄漏。

4. **取消传播**：调用方断开时向 ``_cancelled`` 集合登记 request_id，
   批执行器在下一次 yield 前检查并提前退出，避免为已断开的客户端继续消耗 GPU。
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from ..backends.base import GenerateParams, InferenceBackend, TokenChunk
from ..gateway.schemas import ChatMessage
from ..observability import metrics
from ..observability.logging import get_logger, request_id_var, tenant_var
from .batching import BatchPlan
from .queue import AdmissionQueue, QueuedRequest, QueueFullError
from .scheduler import Scheduler

log = get_logger(__name__)

_SENTINEL = object()  # 队列结束标记


@dataclass(slots=True)
class EngineStats:
    submitted: int = 0
    completed: int = 0
    failed: int = 0
    cancelled: int = 0
    rejected: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "submitted": self.submitted,
            "completed": self.completed,
            "failed": self.failed,
            "cancelled": self.cancelled,
            "rejected": self.rejected,
        }


class RequestHandle:
    """调用方持有的请求句柄：一个私有 chunk 队列 + 取消开关。"""

    __slots__ = ("_cancelled", "_queue", "item")

    def __init__(self, item: QueuedRequest) -> None:
        self.item = item
        self._queue: asyncio.Queue[TokenChunk | object] = asyncio.Queue()
        self._cancelled = False

    @property
    def request_id(self) -> str:
        return self.item.request_id

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def push(self, chunk: TokenChunk) -> None:
        self._queue.put_nowait(chunk)

    def close(self) -> None:
        self._queue.put_nowait(_SENTINEL)

    def cancel(self) -> None:
        self._cancelled = True
        self.close()

    async def stream(self) -> AsyncIterator[TokenChunk]:
        """消费本请求的 token 流，直到后端结束或被取消。"""
        while True:
            chunk = await self._queue.get()
            if chunk is _SENTINEL:
                return
            assert isinstance(chunk, TokenChunk)
            yield chunk


class BatchingEngine:
    """连续批处理引擎：请求 → 批 → token 流。"""

    def __init__(
        self,
        backend: InferenceBackend,
        plan: BatchPlan,
        queue: AdmissionQueue | None = None,
        scheduler: Scheduler | None = None,
        worker_concurrency: int = 4,
    ) -> None:
        self._backend = backend
        self._plan = plan
        self._queue = queue or AdmissionQueue(
            max_size=plan.max_batch_size * 32,
            max_tokens=plan.max_batch_tokens * 4,
        )
        self._scheduler = scheduler or Scheduler(plan, concurrency=worker_concurrency)
        self._stats = EngineStats()
        self._dispatcher: asyncio.Task[None] | None = None
        self._running = False
        self._handles: dict[str, RequestHandle] = {}

    # ------------------------------------------------------------ 只读属性
    @property
    def plan(self) -> BatchPlan:
        return self._plan

    @property
    def queue(self) -> AdmissionQueue:
        return self._queue

    @property
    def scheduler(self) -> Scheduler:
        return self._scheduler

    @property
    def stats(self) -> EngineStats:
        return self._stats

    def stats_snapshot(self) -> dict[str, Any]:
        """聚合快照：供 ``/v1/system/stats``、Grafana 与 HPA 自定义指标使用。"""
        return {
            "engine": self._stats.as_dict(),
            "queue": {
                "depth": self._queue.depth,
                "capacity_left": self._queue.capacity_left,
                "inflight_tokens": self._queue.inflight_tokens,
                "accepted": self._queue.accepted,
                "rejected": self._queue.rejected,
            },
            "scheduler": self._scheduler.stats.as_dict(),
            "plan": {
                "max_batch_size": self._plan.max_batch_size,
                "max_batch_tokens": self._plan.max_batch_tokens,
                "max_wait_ms": self._plan.max_wait_ms,
                "bucket_step": self._plan.bucket_step,
            },
            "backend": self._backend.name,
            "running": self._running,
        }

    # ------------------------------------------------------------ 生命周期
    async def start(self) -> None:
        """启动分发循环并预热后端。"""
        if self._running:
            return
        self._running = True
        await self._backend.warmup()
        self._dispatcher = asyncio.create_task(self._dispatch_loop(), name="aetherserve-dispatcher")
        log.info("engine_started", backend=self._backend.name, plan=str(self._plan))

    async def stop(self, drain: bool = True, timeout_s: float = 30.0) -> None:
        """优雅停机：停止接收新请求，等待在途请求完成后退出。

        顺序很重要：
        1. 停止接受新请求（``_running = False``）—— 但分发循环会继续把
           **已准入**的请求派发完；
        2. 等待队列排空（可选，带超时）；
        3. 关闭队列并终止分发循环；
        4. 释放后端资源（显存、连接池）。

        为什么要"先置位、后关队列"而不是反过来？
        关闭队列会让正在等待的 ``get()`` 立即返回空，分发循环随之退出，
        队列里已准入的请求就再也没人处理了 —— 那不是优雅停机，是静默丢请求。
        """
        self._running = False

        if drain:
            # 这里等待的是"队列是否排空"这一**时间条件**（在 timeout 内持续观察是否变空）。
            # asyncio.Event 只能表达"某事件已发生"，无法表达"等待期内条件保持成立"，
            # 因此使用有界轮询：每 20ms 检查一次，最多 timeout_s 秒，开销可忽略。
            deadline = time.monotonic() + timeout_s
            while self._queue.depth > 0 and time.monotonic() < deadline:  # noqa: ASYNC110
                await asyncio.sleep(0.02)
            if self._queue.depth > 0:  # pragma: no cover - 仅在极端拥塞时出现
                log.warning("stop_drain_timeout", remaining=self._queue.depth, timeout_s=timeout_s)

        await self._queue.close()
        if self._dispatcher is not None:
            try:
                await asyncio.wait_for(self._dispatcher, timeout=5.0)
            except TimeoutError:
                self._dispatcher.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._dispatcher
            except asyncio.CancelledError:  # pragma: no cover
                pass
            self._dispatcher = None

        await self._backend.aclose()
        self._scheduler.log_summary()
        log.info("engine_stopped", **self._stats.as_dict())

    # ------------------------------------------------------------ 提交
    async def submit(
        self,
        model: str,
        tenant: str,
        messages: list[ChatMessage],
        params: GenerateParams,
        priority: int = 0,
        request_id: str | None = None,
    ) -> RequestHandle:
        """准入并提交一个请求。

        Raises:
            QueueFullError: 触发背压（网关应转换为 429）。
        """
        rid = request_id or f"req-{id(object()):x}"
        prompt_tokens = self._backend.estimate_prompt_tokens(messages)
        est_total = prompt_tokens + max(params.max_tokens, 1)

        item = QueuedRequest(
            request_id=rid,
            model=model,
            tenant=tenant,
            messages=messages,
            params=params,
            prompt_tokens=prompt_tokens,
            est_total_tokens=est_total,
            priority=priority,
        )
        try:
            await self._queue.put(item)
        except QueueFullError:
            self._stats.rejected += 1
            raise

        handle = RequestHandle(item)
        self._handles[rid] = handle
        self._stats.submitted += 1
        return handle

    async def cancel(self, request_id: str) -> bool:
        """取消请求（客户端断开 / 超时）。

        必须**同时**设置两个位置：
        - ``handle._cancelled``：让**正在执行**的批执行器在下一次 yield 前退出；
        - ``item.cancelled``：让**仍在准入队列中**的请求被 ``_drain`` 直接丢弃。

        只设前者会导致已排队但未派发的请求照常被执行 —— 白白消耗 GPU 算力。
        """
        handle = self._handles.get(request_id)
        if handle is None:
            return await self._queue.cancel(request_id)

        handle.cancel()
        handle.item.cancelled = True
        # 唤醒分发循环，让它立刻在 _drain 中丢弃这个已取消的请求
        await self._queue.wake()
        self._stats.cancelled += 1
        return True

    # ------------------------------------------------------------ 分发循环
    async def _dispatch_loop(self) -> None:
        """主循环：不断从队列取出批次并派发给执行器。

        这是唯一读取队列的地方，因此批构建策略的调整只需改这一个函数。

        退出条件（两个都为真才退出）：
        1. 已收到停机信号（``_running`` 为假）；
        2. 队列中没有剩余工作，且队列已关闭。

        为什么不能"停机信号一到就退出"？
        因为队列里可能还有**已准入但未派发**的请求 —— 那正是"优雅停机"要照顾的对象。
        直接退出会把它们静默丢弃，调用方的流永远读不到结束标记。

        ``await asyncio.sleep(0)``：``AdmissionQueue.get`` 在"队列已关闭"或
        "仅有被取消请求"时会**立即**返回空列表，此时必须主动让出事件循环，
        否则本循环会变成忙等（busy-spin），把事件循环饿死。
        """
        while True:
            try:
                no_work_left = self._queue.closed and self._queue.depth == 0
                if not self._running and no_work_left:
                    break

                plan = self._scheduler.effective_plan(self._queue.inflight_tokens)
                timeout = 0.1 if not self._running else 0.5
                batch = await self._queue.get(plan, timeout_s=timeout)
                if not batch:
                    if self._queue.closed and self._queue.depth == 0:
                        break
                    await asyncio.sleep(0)
                    continue

                # 派发为独立任务：不同批次之间并行执行（受 scheduler 信号量约束）
                task = asyncio.create_task(self._run_batch(batch), name=f"batch-{batch[0].request_id}")
                task.add_done_callback(self._on_batch_done)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pragma: no cover - 防御性兜底
                log.error("dispatch_loop_error", error=str(exc), exc_info=True)
                await asyncio.sleep(0.05)

    @staticmethod
    def _on_batch_done(task: asyncio.Task[None]) -> None:
        """回收批任务：显式取异常，避免 "Task exception was never retrieved" 噪声。"""
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:  # pragma: no cover
            log.error("batch_task_failed", error=str(exc), exc_info=exc)

    # ------------------------------------------------------------ 批执行
    async def _run_batch(self, batch: list[QueuedRequest]) -> None:
        """执行一个批次，并把 token 分发到各请求的私有队列。"""
        async with self._scheduler:
            tokens = sum(i.est_total_tokens for i in batch)
            await self._scheduler.on_batch_start(len(batch), tokens)
            ok = False
            try:
                await asyncio.gather(
                    *(self._run_one(item) for item in batch if not item.cancelled),
                    return_exceptions=True,
                )
                ok = True
            except Exception as exc:  # pragma: no cover
                log.error("batch_execution_error", error=str(exc), exc_info=True)
            finally:
                await self._scheduler.on_batch_end(ok)
                for item in batch:
                    handle = self._handles.pop(item.request_id, None)
                    if handle is not None:
                        handle.close()

    async def _run_one(self, item: QueuedRequest) -> None:
        """执行批内单个请求：消费后端流并推送到该请求的私有队列。"""
        handle = self._handles.get(item.request_id)
        if handle is None:
            return

        request_id_var.set(item.request_id)
        tenant_var.set(item.tenant)

        token = metrics.REQUESTS_TOTAL.labels(model=item.model, tenant=item.tenant, status="started")
        token.inc()

        t_start = time.monotonic()
        t_first: float | None = None
        t_last: float | None = None
        completion_tokens = 0
        prompt_tokens = item.prompt_tokens
        status = "success"
        backend = self._backend.name

        try:
            async for chunk in self._backend.generate(item.request_id, item.messages, item.params):
                if handle.cancelled:
                    status = "cancelled"
                    break

                now = time.monotonic()
                if t_first is None and chunk.text:
                    t_first = now
                    metrics.TTFT.labels(model=item.model).observe(now - t_start)
                elif t_last is not None and chunk.text:
                    metrics.TPOT.labels(model=item.model).observe(now - t_last)
                if chunk.text:
                    t_last = now

                if chunk.prompt_tokens:
                    prompt_tokens = chunk.prompt_tokens
                if chunk.completion_tokens:
                    completion_tokens = chunk.completion_tokens
                if chunk.finish_reason:
                    handle.push(TokenChunk(text="", finish_reason=chunk.finish_reason))
                    break
                if chunk.text:
                    handle.push(chunk)

            metrics.TOKENS_TOTAL.labels(model=item.model, tenant=item.tenant, kind="prompt").inc(
                prompt_tokens
            )
            metrics.TOKENS_TOTAL.labels(model=item.model, tenant=item.tenant, kind="completion").inc(
                completion_tokens
            )
            metrics.REQUEST_LATENCY.labels(model=item.model, stream="true").observe(
                time.monotonic() - t_start
            )
            self._stats.completed += 1 if status == "success" else 0
            if status == "cancelled":
                self._stats.cancelled += 1

        except asyncio.CancelledError:
            status = "cancelled"
            raise
        except Exception as exc:
            status = "error"
            self._stats.failed += 1
            metrics.BACKEND_ERRORS.labels(backend=backend, kind=type(exc).__name__).inc()
            log.error("request_failed", request_id=item.request_id, error=str(exc), exc_info=True)
            with contextlib.suppress(Exception):
                handle.push(TokenChunk(text="", finish_reason="stop"))
        finally:
            request_id_var.set("-")
            tenant_var.set("-")

        metrics.REQUESTS_TOTAL.labels(model=item.model, tenant=item.tenant, status=status).inc()


# ------------------------------------------------------------------ 便捷构造
def build_engine(settings: Any, backend: InferenceBackend) -> BatchingEngine:
    """按配置构造引擎（网关与测试共用的装配入口）。"""
    plan = BatchPlan.from_settings(settings.batching)
    queue = AdmissionQueue(
        max_size=settings.batching.max_queue_size,
        max_tokens=settings.batching.max_queue_tokens,
        aging_ms=settings.batching.aging_ms,
    )
    scheduler = Scheduler(
        plan,
        concurrency=settings.batching.worker_concurrency,
        kv_budget_tokens=settings.batching.max_queue_tokens,
    )
    return BatchingEngine(
        backend=backend,
        plan=plan,
        queue=queue,
        scheduler=scheduler,
        worker_concurrency=settings.batching.worker_concurrency,
    )


__all__ = ["BatchingEngine", "EngineStats", "RequestHandle", "build_engine"]
