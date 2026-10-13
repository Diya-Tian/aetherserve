"""vLLM 后端。

部署形态说明（面试常被追问）
---------------------------
本后端把 vLLM 作为**独立推理进程/Pod** 使用，网关通过 OpenAI 兼容 HTTP 接口调用它。
这与 Kubernetes 部署模型是一致的：

    [Gateway Pod (N 副本, 无状态)] ──HTTP/SSE──▶ [vLLM Pod (1 副本, 绑定 GPU)]

为什么不把 vLLM 嵌进网关进程？
- 网关需要水平扩容（多副本），而 GPU 卡位有限，两者扩缩容维度不同，耦合会导致副本数被 GPU 绑死；
- 推理引擎崩溃不应让控制面/协议层一起崩；
- 网关可以同时路由到多个 vLLM 实例（不同模型、不同量化），实现模型热切换与灰度。

代价是引入一次进程间网络跳转（本机回环 ~0.1ms，跨节点 ~0.3ms），
相对推理本身几十毫秒的开销可以忽略，且换来清晰的故障域与独立伸缩能力。
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from ..gateway.schemas import ChatMessage
from ..observability import metrics
from ..observability.logging import get_logger
from .base import GenerateParams, InferenceBackend, TokenChunk

log = get_logger(__name__)


class VLLMBackend(InferenceBackend):
    """通过 OpenAI 兼容协议驱动 vLLM / 兼容推理服务。"""

    name = "vllm"

    def __init__(
        self,
        endpoint: str | None = None,
        model: str = "Qwen/Qwen2.5-1.5B-Instruct",
        model_path: str | None = None,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 4096,
        quantization: str | None = None,
        timeout_s: float = 300.0,
        max_connections: int = 256,
    ) -> None:
        self._endpoint = (endpoint or "http://127.0.0.1:8001").rstrip("/")
        self._model = model or model_path or "default"
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.quantization = quantization
        # 连接池：复用 TCP 连接，避免每请求三次握手带来的额外延迟
        limits = httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_connections // 2)
        self._client = httpx.AsyncClient(
            base_url=self._endpoint,
            timeout=httpx.Timeout(timeout_s, connect=10.0),
            limits=limits,
            headers={"User-Agent": "aetherserve/0.1"},
        )
        self._ready = False

    # ------------------------------------------------------------ 协议转换
    @staticmethod
    def _to_openai_messages(messages: list[ChatMessage]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for m in messages:
            content = m.content if isinstance(m.content, str) else json.dumps(m.content, ensure_ascii=False)
            out.append({"role": m.role, "content": content})
        return out

    # ------------------------------------------------------------ 生成
    async def generate(
        self,
        request_id: str,
        messages: list[ChatMessage],
        params: GenerateParams,
    ) -> AsyncIterator[TokenChunk]:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": self._to_openai_messages(messages),
            "max_tokens": params.max_tokens,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if params.stop:
            payload["stop"] = params.stop
        if params.seed is not None:
            payload["seed"] = params.seed

        prompt_tokens = 0
        completion_tokens = 0

        try:
            async with self._client.stream("POST", "/v1/chat/completions", json=payload) as resp:
                if resp.status_code >= 400:
                    body = (await resp.aread()).decode("utf-8", "ignore")[:500]
                    metrics.BACKEND_ERRORS.labels(backend=self.name, kind=f"http_{resp.status_code}").inc()
                    raise RuntimeError(f"vLLM 返回 {resp.status_code}: {body}")

                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    usage = chunk.get("usage") or {}
                    if usage:
                        prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                        completion_tokens = usage.get("completion_tokens", completion_tokens)

                    for choice in chunk.get("choices", []):
                        delta = choice.get("delta") or {}
                        text = delta.get("content") or ""
                        finish = choice.get("finish_reason")
                        if text:
                            completion_tokens += 1
                            yield TokenChunk(
                                text=text,
                                prompt_tokens=prompt_tokens,
                                completion_tokens=completion_tokens,
                            )
                        if finish:
                            yield TokenChunk(
                                text="",
                                finish_reason="length" if finish == "length" else "stop",
                                prompt_tokens=prompt_tokens,
                                completion_tokens=completion_tokens,
                            )
                            return

            yield TokenChunk(
                text="",
                finish_reason="stop",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        except httpx.HTTPError as exc:
            metrics.BACKEND_ERRORS.labels(backend=self.name, kind=type(exc).__name__).inc()
            log.error("vllm_request_error", request_id=request_id, error=str(exc))
            raise

    # ------------------------------------------------------------ 生命周期
    async def warmup(self) -> None:
        """探测后端可用性，并触发一次真实推理完成 CUDA graph / 显存池预热。"""
        try:
            resp = await self._client.get("/v1/models", timeout=10.0)
            self._ready = resp.status_code == 200
        except httpx.HTTPError as exc:
            self._ready = False
            log.warning("vllm_not_ready", endpoint=self._endpoint, error=str(exc))
            return

        if self._ready:
            try:
                async for _ in self.generate(
                    "warmup", [ChatMessage(role="user", content="hi")], GenerateParams(max_tokens=1)
                ):
                    break
                log.info("vllm_warmed_up", model=self._model, endpoint=self._endpoint)
            except Exception as exc:  # pragma: no cover - 预热失败不阻塞启动
                log.warning("vllm_warmup_failed", error=str(exc))

    async def health(self) -> dict[str, Any]:
        try:
            resp = await self._client.get("/v1/models", timeout=5.0)
            ready = resp.status_code == 200
        except httpx.HTTPError:
            ready = False
        return {
            "backend": self.name,
            "ready": ready,
            "endpoint": self._endpoint,
            "model": self._model,
            "quantization": self.quantization,
            "max_model_len": self.max_model_len,
            "gpu_memory_utilization": self.gpu_memory_utilization,
        }

    async def aclose(self) -> None:
        await self._client.aclose()
