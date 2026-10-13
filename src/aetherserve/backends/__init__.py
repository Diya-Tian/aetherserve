"""推理后端插件包。

注意：这里的导入图必须保持"单向"：
``backends.base`` 依赖 ``gateway.schemas``（协议模型），
而 ``gateway.app`` 依赖 ``backends``（装配）。
因此 ``gateway.schemas`` 不能反向依赖任何 backends 模块。
"""

from .base import GenerateParams, InferenceBackend, TokenChunk
from .echo_backend import EchoBackend
from .registry import available_backends, create_backend

__all__ = [
    "EchoBackend",
    "GenerateParams",
    "InferenceBackend",
    "TokenChunk",
    "available_backends",
    "create_backend",
]
