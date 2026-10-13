"""OpenAI 兼容推理网关包。

注意：本模块**不主动导入** ``app`` 子模块。
原因是依赖方向为 ``backends → gateway.schemas``，
若这里在导入时就去加载 ``app``（而 ``app`` 又反向依赖 ``backends``），
就会形成循环导入。改为按需惰性导出，让 ``schemas`` 可以被安全地单独引用。
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # 仅类型检查期可见，运行时零成本
    from .app import GatewayState, create_app

__all__ = ["GatewayState", "create_app"]


def __getattr__(name: str) -> Any:
    """PEP 562 模块级惰性属性：真正用到时才导入重量级子模块。"""
    if name in {"GatewayState", "create_app"}:
        from . import app as _app

        return getattr(_app, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
