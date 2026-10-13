"""OpenAI 协议兼容的数据模型。

只实现推理服务真正需要的最小子集，并对客户端未知字段保持宽容（``extra="allow"``），
以降低业务方从官方 SDK 迁移的成本。
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Role = Literal["system", "user", "assistant", "tool"]
FinishReason = Literal["stop", "length", "content_filter", "tool_calls"]


def new_id(prefix: str) -> str:
    """生成带前缀的请求 ID，便于日志与链路追踪关联。"""
    return f"{prefix}-{uuid.uuid4().hex[:24]}"


def now_ts() -> int:
    return int(time.time())


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: Role
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None


class ChatCompletionRequest(BaseModel):
    """``POST /v1/chat/completions`` 请求体。"""

    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[ChatMessage]
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    max_tokens: int | None = Field(default=None, ge=1)
    stop: str | list[str] | None = None
    stream: bool = False
    stream_options: dict[str, Any] | None = None
    presence_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    frequency_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    user: str | None = None
    seed: int | None = None
    n: int = Field(default=1, ge=1, le=8)

    @field_validator("messages")
    @classmethod
    def _non_empty(cls, v: list[ChatMessage]) -> list[ChatMessage]:
        if not v:
            raise ValueError("messages 不能为空")
        return v


class CompletionRequest(BaseModel):
    """``POST /v1/completions`` 请求体（兼容旧接口）。"""

    model_config = ConfigDict(extra="allow")

    model: str
    prompt: str | list[str]
    max_tokens: int | None = Field(default=None, ge=1)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    stop: str | list[str] | None = None
    stream: bool = False


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: FinishReason | None = None
    logprobs: dict[str, Any] | None = None


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: new_id("chatcmpl"))
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=now_ts)
    model: str = ""
    choices: list[ChatCompletionChoice] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    system_fingerprint: str | None = None


class ChatCompletionChunkDelta(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: Role | None = None
    content: str | None = None


class ChatCompletionChunkChoice(BaseModel):
    index: int = 0
    delta: ChatCompletionChunkDelta = Field(default_factory=ChatCompletionChunkDelta)
    finish_reason: FinishReason | None = None


class ChatCompletionChunk(BaseModel):
    """SSE 流式返回的单片。"""

    id: str = Field(default_factory=lambda: new_id("chatcmpl"))
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=now_ts)
    model: str = ""
    choices: list[ChatCompletionChunkChoice] = Field(default_factory=list)
    usage: Usage | None = None


class ModelCard(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int = Field(default_factory=now_ts)
    owned_by: str = "aetherserve"


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelCard] = Field(default_factory=list)


class ErrorDetail(BaseModel):
    message: str
    type: str
    code: str | None = None
    param: str | None = None


class ErrorResponse(BaseModel):
    """OpenAI 风格错误体，保证客户端 SDK 能正确解析。"""

    error: ErrorDetail
