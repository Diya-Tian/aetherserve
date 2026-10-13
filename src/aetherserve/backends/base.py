"""推理后端插件化抽象层。

设计权衡
--------
后端被抽象成"**执行器**"而非"服务器"：它只负责把一批请求变成 token 流，
**不负责**准入控制、配额、路由与伸缩信号 —— 那些属于可独立演进的网关/引擎层。
这样 vLLM / Transformers / Echo 可以互换，也方便在 CI 中用假后端做确定性测试。
"""

from __future__ import annotations

import abc
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from ..gateway.schemas import ChatMessage


@dataclass(slots=True)
class GenerateParams:
    """一次生成请求的采样参数（与后端无关的统一表示）。"""

    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 1.0
    stop: list[str] = field(default_factory=list)
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    seed: int | None = None


@dataclass(slots=True)
class TokenChunk:
    """后端吐出的一个增量片段。"""

    text: str
    token_ids: list[int] = field(default_factory=list)
    finish_reason: str | None = None  # stop | length | None
    prompt_tokens: int = 0
    completion_tokens: int = 0


class InferenceBackend(abc.ABC):
    """所有推理后端的统一契约。"""

    name: str = "base"

    @abc.abstractmethod
    async def generate(
        self,
        request_id: str,
        messages: list[ChatMessage],
        params: GenerateParams,
    ) -> AsyncIterator[TokenChunk]:
        """流式生成。实现方必须以 ``yield`` 返回增量，最后一块可带 ``finish_reason``。"""
        raise NotImplementedError
        yield TokenChunk(text="")  # pragma: no cover - 让类型检查器识别为异步生成器

    async def warmup(self) -> None:
        """预热：加载权重、编译图、填充显存池。Operator 用它做 readiness 门禁。"""
        return None

    async def health(self) -> dict[str, Any]:
        """返回后端健康状态，供 ``/healthz`` 与 Operator 探针使用。"""
        return {"backend": self.name, "ready": True}

    async def aclose(self) -> None:
        """释放资源（显存、进程、连接池）。"""
        return None

    @staticmethod
    def estimate_prompt_tokens(messages: list[ChatMessage]) -> int:
        """粗略估算 prompt token 数，用于准入控制与 KV Cache 预算。

        真实分词依赖模型词表，这里用"中文≈1 token/字、英文≈1 token/4 字符"的启发式，
        误差在显存预算场景下可接受（预算本身留有安全水位）。
        """
        total: float = 0.0
        for m in messages:
            content = m.content if isinstance(m.content, str) else str(m.content or "")
            for ch in content:
                # 中文（含 CJK 标点）按 1 token/字计；英文按 ~1 token/4 字符计
                total += 1.0 if ord(ch) > 0x2E80 else 0.25
            total += 4  # 角色与分隔符开销
        return int(total) + 8
