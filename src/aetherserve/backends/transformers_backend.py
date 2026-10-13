"""HuggingFace Transformers 后端（单机/CPU 兼容路径）。

定位
----
vLLM 是生产路径，但它需要 Linux + CUDA + 较新的 GPU。
Transformers 后端用于：Windows/CPU 开发机、小模型（<1B）快速验证、以及**算法对照实验**
（例如验证自研批处理策略相对逐请求推理的收益）。

实现要点
--------
用 ``TextIteratorStreamer`` + 独立线程做增量解码：
模型前向必须在调用线程外执行（否则阻塞事件循环），
streamer 提供线程安全的阻塞队列，事件循环侧用 ``asyncio.to_thread`` 读取，
从而在不引入新的并发框架的前提下把同步推理接入 asyncio。

设计取舍：这里按"文本增量"而非"逐 token"回传。
原因是 Transformers 的 streamer 输出经过 tokenizer 解码后可能一次给出多个字符，
强行按单字切分会破坏 UTF-8 边界与数字/英文单词的完整性。
语义上仍然是流式（首块到达即触发 TTFT），对用户体验无影响。
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator
from typing import Any

from ..gateway.schemas import ChatMessage
from ..observability import metrics
from ..observability.logging import get_logger
from .base import GenerateParams, InferenceBackend, TokenChunk

log = get_logger(__name__)


class TransformersBackend(InferenceBackend):
    """基于 HuggingFace Transformers 的本地推理后端。"""

    name = "transformers"

    def __init__(
        self,
        model_path: str = "Qwen/Qwen2.5-0.5B-Instruct",
        max_model_len: int = 4096,
        device: str | None = None,
        dtype: str = "auto",
    ) -> None:
        self._model_path = model_path
        self.max_model_len = max_model_len
        self._device = device
        self._dtype = dtype
        self._model: Any = None
        self._tokenizer: Any = None
        self._lock = threading.Lock()  # 同一模型实例不支持并发前向

    # ------------------------------------------------------------ 加载
    def _ensure_loaded(self) -> None:
        """惰性加载权重：避免 import 时就吃掉显存（也让 CI 可跳过）。"""
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            import torch  # 延迟导入：默认安装不含 torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            log.info("loading_model", model=self._model_path)
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_path, trust_remote_code=True)

            kwargs: dict[str, Any] = {"trust_remote_code": True}
            if self._dtype != "auto":
                kwargs["torch_dtype"] = getattr(torch, self._dtype)
            else:
                kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32

            self._model = AutoModelForCausalLM.from_pretrained(self._model_path, **kwargs)
            if self._device:
                self._model = self._model.to(self._device)
            elif torch.cuda.is_available():
                self._model = self._model.to("cuda")
            self._model.eval()
            log.info("model_loaded", model=self._model_path, device=str(getattr(self._model, "device", "?")))

    # ------------------------------------------------------------ 生成
    async def generate(
        self,
        request_id: str,
        messages: list[ChatMessage],
        params: GenerateParams,
    ) -> AsyncIterator[TokenChunk]:
        self._ensure_loaded()

        from transformers import TextIteratorStreamer

        chat = [{"role": m.role, "content": m.content or ""} for m in messages]
        try:
            prompt = self._tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            inputs = self._tokenizer([prompt], return_tensors="pt")
        except Exception:
            # 非 chat 模板模型回退为纯文本拼接
            prompt = "\n".join(f"{m.role}: {m.content or ''}" for m in messages) + "\nassistant:"
            inputs = self._tokenizer([prompt], return_tensors="pt")

        device = getattr(self._model, "device", "cpu")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        prompt_tokens = int(inputs["input_ids"].shape[-1])

        streamer = TextIteratorStreamer(
            self._tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=300.0
        )
        gen_kwargs: dict[str, Any] = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": params.max_tokens,
            "do_sample": params.temperature > 0,
            "temperature": max(params.temperature, 1e-5),
            "top_p": params.top_p,
            "pad_token_id": self._tokenizer.pad_token_id or self._tokenizer.eos_token_id,
        }

        thread = threading.Thread(target=self._model.generate, kwargs=gen_kwargs, daemon=True)
        thread.start()

        completion_tokens = 0
        try:
            while True:
                try:
                    piece = await asyncio.to_thread(next, streamer)
                except StopIteration:
                    break
                except Exception as exc:  # pragma: no cover
                    metrics.BACKEND_ERRORS.labels(backend=self.name, kind=type(exc).__name__).inc()
                    log.error("transformers_stream_error", request_id=request_id, error=str(exc))
                    raise

                if not piece:
                    continue
                completion_tokens += max(1, len(piece))
                yield TokenChunk(
                    text=piece,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
        finally:
            # 保证生成线程收敛，避免 daemon 线程持有显存引用
            thread.join(timeout=1.0)

        yield TokenChunk(
            text="",
            finish_reason="length" if completion_tokens >= params.max_tokens else "stop",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    # ------------------------------------------------------------ 生命周期
    async def warmup(self) -> None:
        try:
            await asyncio.to_thread(self._ensure_loaded)
        except Exception as exc:  # pragma: no cover
            log.warning("transformers_warmup_failed", error=str(exc))

    async def health(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "ready": self._model is not None,
            "model": self._model_path,
            "loaded": self._model is not None,
        }

    async def aclose(self) -> None:
        self._model = None
        self._tokenizer = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
