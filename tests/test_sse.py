"""SSE 编码与心跳测试。"""

from __future__ import annotations

import asyncio

from aetherserve.gateway.sse import (
    SSE_DONE,
    SSE_HEADERS,
    heartbeat_stream,
    sse_comment,
    sse_pack,
    sse_retry,
)


class TestPack:
    def test_json_frame_is_single_line(self) -> None:
        """SSE 帧必须是单行，否则客户端会把 JSON 拆成无效的多条 data。"""
        frame = sse_pack({"text": "第一行\n第二行"})
        assert frame.startswith("data: ")
        assert frame.endswith("\n\n")
        body = frame[len("data: ") : -2]
        assert "\n" not in body, "帧体出现了换行，会破坏 SSE 协议"
        # 换行必须被转义而不是原样输出
        assert "\\n" in body

    def test_bytes_payload(self) -> None:
        assert sse_pack(b'{"a":1}') == 'data: {"a":1}\n\n'

    def test_str_payload(self) -> None:
        assert sse_pack("hello") == "data: hello\n\n"

    def test_done_constant(self) -> None:
        assert SSE_DONE == "data: [DONE]\n\n"

    def test_headers_disable_buffering(self) -> None:
        assert SSE_HEADERS["X-Accel-Buffering"] == "no"
        assert "no-cache" in SSE_HEADERS["Cache-Control"]
        assert SSE_HEADERS["Content-Type"].startswith("text/event-stream")

    def test_comment_and_retry(self) -> None:
        assert sse_comment().startswith(": ")
        assert sse_retry(3000) == "retry: 3000\n\n"


class TestHeartbeat:
    async def test_passes_through_events(self) -> None:
        async def source():
            for i in range(3):
                yield f"data: {i}\n\n"

        got = [x async for x in heartbeat_stream(source(), interval_s=10.0)]
        assert got == ["data: 0\n\n", "data: 1\n\n", "data: 2\n\n"]

    async def test_emits_heartbeat_when_idle(self) -> None:
        """空闲超过间隔时必须发送注释心跳，防止中间设备断连。

        事件源用"队列驱动的异步生成器"，与真实链路（引擎的 chunk 队列）同构，
        因此本用例也顺带验证了心跳不会吞掉或重排真实事件。
        """
        events: asyncio.Queue[str | None] = asyncio.Queue()

        async def source():
            while True:
                item = await events.get()
                if item is None:
                    return
                yield item

        frames: list[str] = []

        async def consume() -> None:
            async for frame in heartbeat_stream(source(), interval_s=0.05):
                frames.append(frame)
                if frame == "data: second\n\n":
                    await events.put(None)

        await events.put("data: first\n\n")
        task = asyncio.create_task(consume())
        await asyncio.sleep(0.2)  # 让心跳跑几个周期
        await events.put("data: second\n\n")
        await asyncio.wait_for(task, timeout=5.0)

        pings = [f for f in frames if f.startswith(": ")]
        assert pings, f"未发送心跳: {frames}"
        assert frames[0] == "data: first\n\n"
        # 真实事件必须原序完整送达，且心跳不得出现在末尾（[DONE] 之后不能有 ping）
        assert [f for f in frames if not f.startswith(": ")] == ["data: first\n\n", "data: second\n\n"]
        assert frames[-1] == "data: second\n\n", f"结束帧后仍插入心跳: {frames[-3:]}"

    async def test_no_ping_after_final_event(self) -> None:
        """回归用例：事件流正常结束时不允许多余心跳（超时与入队的竞态）。"""

        async def source():
            yield "data: only\n\n"

        got = [x async for x in heartbeat_stream(source(), interval_s=0.001)]
        assert got == ["data: only\n\n"]

    async def test_empty_source(self) -> None:
        async def empty():
            return
            yield  # pragma: no cover

        got = [x async for x in heartbeat_stream(empty(), interval_s=0.01)]
        assert got == []
