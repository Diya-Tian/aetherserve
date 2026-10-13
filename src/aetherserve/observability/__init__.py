"""``observability`` 包：Prometheus 指标、结构化日志、链路追踪。"""

from .metrics import REGISTRY, render_metrics

__all__ = ["REGISTRY", "render_metrics"]
