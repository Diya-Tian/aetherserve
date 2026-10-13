"""Echo 后端：确定性假推理后端。

用途
----
1. **CI / 单元测试**：不依赖 GPU 与模型权重，令牌流完全可预测；
2. **压测对照组**：把"网络与调度开销"与"GPU 计算开销"解耦，
   从而量化网关层自身在 P99 上的占比（即"网关引入多少额外延迟"）。
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import AsyncIterator
from typing import Any

from ..gateway.schemas import ChatMessage
from .base import GenerateParams, InferenceBackend, TokenChunk

# 一个足够小的"词表"，让输出看起来像自然语言，同时保持确定性
_VOCAB = list("一二三四五六七八九十的了吗呢啊是在有和就不人都一个上也很到说要去你会着没有看好自己这")


class EchoBackend(InferenceBackend):
    """按 token 逐个回显，模拟真实解码节奏。"""

    name = "echo"

    def __init__(self, token_delay_ms: float = 2.0, max_tokens_hint: int = 64) -> None:
        self._token_delay = token_delay_ms / 1000.0
        self._max_tokens_hint = max_tokens_hint

    async def generate(
        self,
        request_id: str,
        messages: list[ChatMessage],
        params: GenerateParams,
    ) -> AsyncIterator[TokenChunk]:
        prompt_tokens = self.estimate_prompt_tokens(messages)
        # 用 request_id 派生的确定性序列，保证同请求可复现，便于断言
        seed = int(hashlib.sha256(request_id.encode()).hexdigest()[:8], 16)
        n = min(params.max_tokens, self._max_tokens_hint)
        produced = 0

        for i in range(n):
            text = _VOCAB[(seed + i) % len(_VOCAB)]
            produced += 1
            if self._token_delay:
                await asyncio.sleep(self._token_delay)
            yield TokenChunk(
                text=text,
                token_ids=[(seed + i) % 30000],
                prompt_tokens=prompt_tokens,
                completion_tokens=produced,
            )

        yield TokenChunk(
            text="",
            finish_reason="length" if produced >= params.max_tokens else "stop",
            prompt_tokens=prompt_tokens,
            completion_tokens=produced,
        )

    async def health(self) -> dict[str, Any]:
        return {"backend": self.name, "ready": True, "deterministic": True}
