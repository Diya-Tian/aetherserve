"""推理后端注册表：按名称构造后端实例。

扩展方式（对使用者友好）
------------------------
新增一个推理后端只需三步：
1. 继承 ``InferenceBackend`` 并实现 ``generate``；
2. 在 ``_FACTORIES`` 注册名称；
3. 在 ``pyproject.toml`` 里把重依赖放进可选依赖组，避免拖累默认安装。

这样主程序的依赖保持轻量（CI 无需安装 torch/vllm），
也说明模块边界是"依赖倒置"的：高层策略不依赖低层实现细节。
"""

from __future__ import annotations

from typing import Any

from .base import GenerateParams, InferenceBackend, TokenChunk
from .echo_backend import EchoBackend

__all__ = [
    "EchoBackend",
    "GenerateParams",
    "InferenceBackend",
    "TokenChunk",
    "available_backends",
    "create_backend",
]


def _make_transformers(settings: Any) -> InferenceBackend:
    from .transformers_backend import TransformersBackend

    return TransformersBackend(
        model_path=settings.backend.model_path,
        max_model_len=settings.backend.max_model_len,
    )


def _make_vllm(settings: Any) -> InferenceBackend:
    from .vllm_backend import VLLMBackend

    return VLLMBackend(
        endpoint=settings.backend.vllm_endpoint,
        model=settings.backend.model,
        model_path=settings.backend.model_path,
        gpu_memory_utilization=settings.backend.gpu_memory_utilization,
        max_model_len=settings.backend.max_model_len,
        quantization=settings.backend.quantization,
    )


def _make_echo(settings: Any) -> InferenceBackend:
    return EchoBackend()


_FACTORIES: dict[str, Any] = {
    "echo": _make_echo,
    "transformers": _make_transformers,
    "vllm": _make_vllm,
}


def available_backends() -> list[str]:
    """返回已注册的后端名称（供 ``/v1/system/info`` 展示）。"""
    return sorted(_FACTORIES)


def create_backend(settings: Any, name: str | None = None) -> InferenceBackend:
    """按配置或显式名称创建后端。

    Args:
        settings: 全局 ``Settings`` 实例。
        name: 覆盖 ``settings.backend.backend``。

    Raises:
        ValueError: 名称未注册。
    """
    key = (name or settings.backend.backend or "echo").lower()
    factory = _FACTORIES.get(key)
    if factory is None:
        raise ValueError(f"未知后端 {key!r}，可选：{', '.join(available_backends())}")
    return factory(settings)
