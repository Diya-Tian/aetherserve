"""Server-Sent Events (SSE) 流式输出封装。

协议为什么要长这样？
--------------------
OpenAI 的流式协议 = **SSE over HTTP/1.1 分块传输**，格式为：

    data: {"id":"...","choices":[{"delta":{"content":"你"}}]}\\n\\n
    data: [DONE]\\n\\n

工程上必须处理的坑：
1. **禁用中间层缓冲**：必须回 ``X-Accel-Buffering: no`` 与 ``Cache-Control: no-cache``，
   否则 nginx/网关会攒满缓冲才下发，用户会感觉"卡一下然后全出来"（TTFT 指标失真）；
2. **保活心跳**：SSE 连接可能被中间设备按空闲超时切断，需周期性发注释行 ``: ping``；
3. **格式化**：JSON 内含换行会破坏 SSE 帧，因此必须保证单行紧凑序列化（orjson 默认即单行）；
4. **反压**：客户端读得慢时不能让服务端无限缓冲，需配合有界队列。
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Iterable
from typing import Any

import orjson

SSE_DONE = "data: [DONE]\n\n"
SSE_HEADERS: dict[str, str] = {
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",  # 关键：让 nginx 不做响应缓冲
    "Content-Type": "text/event-stream; charset=utf-8",
}

# 队列容量：限制"生产快于消费"时的内存占用（背压的第一个水位）
_HEARTBEAT_QUEUE_SIZE = 64


def sse_pack(payload: Any) -> str:
    """把对象序列化为一个 SSE 帧（单行 JSON）。"""
    if isinstance(payload, (bytes, bytearray)):
        data = bytes(payload)
    elif isinstance(payload, str):
        data = payload.encode("utf-8")
    else:
        data = orjson.dumps(payload)
    return f"data: {data.decode('utf-8')}\n\n"


def sse_comment(text: str = "ping") -> str:
    """SSE 注释行，用于保活（客户端会忽略）。"""
    return f": {text}\n\n"


def sse_retry(ms: int) -> str:
    """告知客户端断线重连间隔。"""
    return f"retry: {ms}\n\n"


async def heartbeat_stream(
    source: AsyncIterator[str],
    interval_s: float = 15.0,
) -> AsyncIterator[str]:
    """在事件流空闲时插入心跳，避免长连接被中间设备回收。

    实现方式：起一个**独立的生产者任务**把 ``source`` 的事件推入有界队列，
    消费侧以 ``interval_s`` 为超时从队列取事件；超时即发一次心跳。

    为什么不用 "``asyncio.wait`` 竞速两个协程" 的写法？
    因为那样要求消费侧每一轮都正确地续跑同一个 ``__anext__`` 任务，
    一旦某轮把任务引用丢掉或错误地取消它，生成器就再也不会被恢复，
    表现为"只发心跳、事件永远不来"——一个非常难排查的静默故障。
    独立生产者任务把"拉取"与"节拍"彻底解耦，语义更清晰，也没有这类竞态。
    """
    queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=_HEARTBEAT_QUEUE_SIZE)

    async def producer() -> None:
        try:
            async for item in source:
                await queue.put(item)
        finally:
            # None 作为结束哨兵：无论正常结束还是被取消都要送达，避免消费侧永久阻塞
            with contextlib.suppress(asyncio.QueueFull):
                queue.put_nowait(None)

    task = asyncio.create_task(producer(), name="sse-heartbeat-producer")
    try:
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=interval_s)
            except TimeoutError:
                # 关键：超时后先尝试"非阻塞排空"，再决定是否发心跳。
                # 否则会出现竞态 —— 哨兵/事件已在超时瞬间入队，
                # 我们却仍插入一个心跳，使客户端在 [DONE] 之后还收到 ping。
                try:
                    item = queue.get_nowait()
                except asyncio.QueueEmpty:
                    yield sse_comment()
                    continue

            if item is None:
                return
            yield item
    finally:
        if not task.done():
            task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task


def join_frames(frames: Iterable[str]) -> str:
    """把多个帧拼成一个字符串（测试与调试用）。"""
    return "".join(frames)
