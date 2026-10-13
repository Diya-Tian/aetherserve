"""结构化日志：输出 JSON，便于 ELK / Loki 采集与按 request_id 检索。

分布式系统中"问题定界定位"的第一步是**请求可追踪**。
我们通过 contextvars 把 request_id / tenant 注入日志上下文，
业务代码无需层层传参即可让全链路日志带上同一 request_id。
"""

from __future__ import annotations

import logging
import sys
from contextvars import ContextVar
from typing import Any

import structlog

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")
tenant_var: ContextVar[str] = ContextVar("tenant", default="-")


def _inject_context(_logger: Any, _method: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    event_dict["request_id"] = request_id_var.get()
    event_dict["tenant"] = tenant_var.get()
    return event_dict


def configure_logging(level: str = "INFO", json_output: bool = True) -> None:
    """初始化 structlog。开发环境可用 ``json_output=False`` 获得彩色可读输出。"""
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=getattr(logging, level.upper(), 20))

    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        _inject_context,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    processors.append(
        structlog.processors.JSONRenderer(ensure_ascii=False)
        if json_output
        else structlog.dev.ConsoleRenderer()
    )

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, level.upper(), 20)),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = "aetherserve") -> Any:
    return structlog.get_logger(name)
