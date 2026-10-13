"""批处理引擎测试：端到端流式、批聚合、错误隔离、取消、停机。"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from aetherserve.backends.base import GenerateParams, InferenceBackend, TokenChunk
from aetherserve.backends.echo_backend import EchoBackend
from aetherserve.config import Settings
from aetherserve.engine import BatchingEngine, build_engine
from aetherserve.engine.batching import BatchPlan
from aetherserve.engine.queue import QueuedRequest
from aetherserve.gateway.schemas import ChatMessage


def user_msg(text: str = "你好") -> list[ChatMessage]:
    return [ChatMessage(role="user", content=text)]


class RecordingBackend(InferenceBackend):
    """记录每次 generate 调用的后端，用于断言批聚合行为。"""

    name = "recording"

    def __init__(self, delay_ms: float = 1.0) -> None:
        self.calls: list[str] = []
        self.concurrent = 0
        self.max_concurrent = 0
        self._delay = delay_ms / 1000.0

    async def generate(
        self, request_id: str, messages: list[ChatMessage], params: GenerateParams
    ) -> AsyncIterator[TokenChunk]:
        self.calls.append(request_id)
        self.concurrent += 1
        self.max_concurrent = max(self.max_concurrent, self.concurrent)
        try:
            for i in range(3):
                if self._delay:
                    await asyncio.sleep(self._delay)
                yield TokenChunk(text=f"{i}", prompt_tokens=5, completion_tokens=i + 1)
            yield TokenChunk(text="", finish_reason="stop", prompt_tokens=5, completion_tokens=3)
        finally:
            self.concurrent -= 1


class FailingBackend(InferenceBackend):
    """一半请求失败，用于验证批内错误隔离。"""

    name = "failing"

    async def generate(
        self, request_id: str, messages: list[ChatMessage], params: GenerateParams
    ) -> AsyncIterator[TokenChunk]:
        if request_id.endswith("bad"):
            raise RuntimeError("模拟推理失败")
        yield TokenChunk(text="ok", prompt_tokens=1, completion_tokens=1)
        yield TokenChunk(text="", finish_reason="stop", prompt_tokens=1, completion_tokens=1)


@pytest.fixture
def engine_settings() -> Settings:
    s = Settings()
    s.batching.max_batch_size = 8
    s.batching.max_batch_tokens = 4096
    s.batching.max_wait_ms = 30.0
    s.batching.worker_concurrency = 2
    s.backend.model = "test-model"
    return s


class TestEngineBasics:
    async def test_submit_and_stream(self, engine: BatchingEngine) -> None:
        handle = await engine.submit("echo-1", "t1", user_msg(), GenerateParams(max_tokens=5))
        tokens: list[str] = []
        finish: str | None = None
        async for chunk in handle.stream():
            if chunk.finish_reason:
                finish = chunk.finish_reason
            if chunk.text:  # 结束帧的 text 为空，不计入 token
                tokens.append(chunk.text)
        assert len(tokens) == 5
        assert finish == "length"

    async def test_stats_after_request(self, engine: BatchingEngine) -> None:
        handle = await engine.submit("echo-1", "t1", user_msg(), GenerateParams(max_tokens=4))
        async for _ in handle.stream():
            pass
        snap = engine.stats_snapshot()
        assert snap["engine"]["submitted"] == 1
        assert snap["engine"]["completed"] == 1
        assert snap["scheduler"]["completed_batches"] >= 1
        assert snap["queue"]["depth"] == 0
        assert snap["queue"]["inflight_tokens"] == 0  # 请求结束后必须归还显存记账

    async def test_prompt_token_estimation(self, engine: BatchingEngine) -> None:
        """中文与英文的 token 估算应有显著差异（启发式分词）。"""
        zh = await engine.submit("m", "t", user_msg("你好世界你好世界"), GenerateParams(max_tokens=1))
        en = await engine.submit("m", "t", user_msg("hello world hello world"), GenerateParams(max_tokens=1))
        assert zh.item.prompt_tokens > en.item.prompt_tokens


class TestBatchAggregation:
    async def test_concurrent_requests_are_batched(self, engine_settings: Settings) -> None:
        """并发提交的请求应被聚合到同一个批（批处理的核心价值）。"""
        backend = RecordingBackend(delay_ms=10.0)
        eng = build_engine(engine_settings, backend)
        await eng.start()
        try:
            handles = [
                await eng.submit("m", "t1", user_msg(f"q{i}"), GenerateParams(max_tokens=3)) for i in range(4)
            ]
            await asyncio.gather(*[self._drain(h) for h in handles])
        finally:
            await eng.stop(drain=False)

        assert len(backend.calls) == 4
        snap = eng.stats_snapshot()
        assert snap["scheduler"]["peak_batch_size"] > 1, (
            f"并发请求未被聚合，peak_batch_size={snap['scheduler']['peak_batch_size']}"
        )

    async def test_dispatcher_respects_concurrency_limit(self, engine_settings: Settings) -> None:
        """批执行器并发度受 worker_concurrency 限制（保护显存）。"""
        engine_settings.batching.worker_concurrency = 1
        engine_settings.batching.max_batch_size = 1
        engine_settings.batching.max_wait_ms = 1.0
        backend = RecordingBackend(delay_ms=5.0)
        eng = build_engine(engine_settings, backend)
        await eng.start()
        try:
            handles = [
                await eng.submit("m", "t1", user_msg(f"q{i}"), GenerateParams(max_tokens=3)) for i in range(5)
            ]
            await asyncio.gather(*[self._drain(h) for h in handles])
        finally:
            await eng.stop(drain=False)

        assert backend.max_concurrent == 1, f"并发度超限: {backend.max_concurrent}"

    @staticmethod
    async def _drain(handle: object) -> None:
        async for _ in handle.stream():  # type: ignore[attr-defined]
            pass


class TestErrorIsolation:
    async def test_one_failure_does_not_break_batch(self) -> None:
        """批内单个请求失败，其他请求必须正常完成（错误隔离）。"""
        s = Settings()
        s.batching.max_batch_size = 8
        s.batching.max_wait_ms = 20.0
        eng = BatchingEngine(
            backend=FailingBackend(),
            plan=BatchPlan(max_batch_size=8, max_batch_tokens=4096, max_wait_ms=20.0),
        )
        await eng.start()
        try:
            good = await eng.submit("m", "t1", user_msg(), GenerateParams(max_tokens=2), request_id="good-1")
            bad = await eng.submit("m", "t1", user_msg(), GenerateParams(max_tokens=2), request_id="bad")
            good_out = await self._collect(good)
            bad_out = await self._collect(bad)
        finally:
            await eng.stop(drain=False)

        assert good_out == ["ok"]
        assert bad_out == []  # 失败请求收到空流（由网关转换为错误响应）
        snap = eng.stats_snapshot()
        assert snap["engine"]["failed"] == 1
        assert snap["scheduler"]["failed_batches"] == 0, "单请求失败不应把整批判定为失败"

    @staticmethod
    async def _collect(handle: object) -> list[str]:
        out = []
        async for chunk in handle.stream():  # type: ignore[attr-defined]
            if chunk.text:
                out.append(chunk.text)
        return out


class TestCancellation:
    async def test_cancel_in_queue_prevents_execution(self, engine_settings: Settings) -> None:
        """排队中的请求被取消后，后端不应再为它执行推理。"""
        engine_settings.batching.max_wait_ms = 200.0
        engine_settings.batching.max_batch_size = 1  # 强制串行，让第二个请求排队
        engine_settings.batching.worker_concurrency = 1
        backend = RecordingBackend(delay_ms=80.0)
        eng = build_engine(engine_settings, backend)
        await eng.start()
        try:
            h1 = await eng.submit(
                "m", "t1", user_msg("first"), GenerateParams(max_tokens=3), request_id="first"
            )
            await asyncio.sleep(0.02)  # 确保 h1 已进入执行
            h2 = await eng.submit(
                "m", "t1", user_msg("second"), GenerateParams(max_tokens=3), request_id="second"
            )
            assert eng.queue.depth == 1, "h2 应处于排队状态"
            assert await eng.cancel(h2.request_id) is True
            await TestBatchAggregation._drain(h1)
            # 等到 h1 整批结束；若取消失效，h2 会在此时被执行
            await asyncio.sleep(0.2)
        finally:
            await eng.stop(drain=False)

        assert "second" not in backend.calls, f"被取消的请求仍被执行: {backend.calls}"
        assert h2.cancelled is True

    async def test_cancelled_request_returns_stream_end(self, engine: BatchingEngine) -> None:
        handle = await engine.submit("m", "t1", user_msg(), GenerateParams(max_tokens=50))
        await engine.cancel(handle.request_id)
        out = [c async for c in handle.stream()]
        assert out == [] or all(not c.text for c in out)

    async def test_cancel_unknown_request(self, engine: BatchingEngine) -> None:
        """取消不存在的请求必须安全返回 False，不能抛异常。"""
        assert await engine.cancel("no-such-request") is False


class TestLifecycle:
    async def test_start_is_idempotent(self, engine: BatchingEngine) -> None:
        await engine.start()  # 重复调用不应创建第二个分发循环
        snap = engine.stats_snapshot()
        assert snap["running"] is True

    async def test_stop_drains_inflight_requests(self) -> None:
        """优雅停机：在途请求应被允许跑完，而不是被切断。

        注意：必须**并发**消费流（先发起消费任务，再调用 stop），
        因为 ``stop()`` 会等待队列排空，若串行等待会死锁。
        """
        s = Settings()
        s.batching.max_wait_ms = 1.0
        s.batching.max_batch_size = 2
        eng = build_engine(s, EchoBackend(token_delay_ms=0.5, max_tokens_hint=5))
        await eng.start()

        handle = await eng.submit("m", "t", user_msg(), GenerateParams(max_tokens=5))
        collected: list[str] = []

        async def consume() -> None:
            async for chunk in handle.stream():
                if chunk.text:
                    collected.append(chunk.text)

        consumer = asyncio.create_task(consume())
        await eng.stop(drain=True, timeout_s=5.0)
        await asyncio.wait_for(consumer, timeout=5.0)

        assert len(collected) == 5, f"在途请求未被完整处理: {collected}"

    async def test_stop_idempotent(self, engine: BatchingEngine) -> None:
        await engine.stop(drain=False)
        await engine.stop(drain=False)


class TestAdmissionThroughEngine:
    async def test_backpressure_returns_error(self, engine_settings: Settings) -> None:
        """队列容量耗尽时必须快速失败（429），而不是无限排队。"""
        engine_settings.batching.max_queue_size = 2
        engine_settings.batching.max_queue_tokens = 100_000
        engine_settings.batching.max_wait_ms = 500.0
        engine_settings.batching.worker_concurrency = 1
        engine_settings.batching.max_batch_size = 1
        eng = build_engine(engine_settings, EchoBackend(token_delay_ms=30.0, max_tokens_hint=10))
        await eng.start()
        from aetherserve.engine.queue import QueueFullError

        try:
            accepted = 0
            for i in range(10):
                try:
                    await eng.submit("m", "t1", user_msg(f"q{i}"), GenerateParams(max_tokens=10))
                    accepted += 1
                except QueueFullError:
                    pass
            assert accepted < 10, "背压未生效，所有请求都被接纳"
            assert eng.stats_snapshot()["queue"]["rejected"] > 0
        finally:
            await eng.stop(drain=False)


class TestQueueInternals:
    async def test_handle_registry_cleaned(self, engine: BatchingEngine) -> None:
        """请求完成后必须从 handle 表移除，否则会内存泄漏。"""
        handle = await engine.submit("m", "t1", user_msg(), GenerateParams(max_tokens=3))
        async for _ in handle.stream():
            pass
        await asyncio.sleep(0.05)
        assert handle.request_id not in engine._handles

    def test_queued_request_waited_ms_grows(self) -> None:
        item = QueuedRequest(
            request_id="x",
            model="m",
            tenant="t",
            messages=[],
            params=None,
            prompt_tokens=1,
            est_total_tokens=1,
        )
        assert item.waited_ms >= 0.0
