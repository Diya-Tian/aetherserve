"""Operator 进程入口测试。

``main.py`` 只做三件事：配置日志、注册处理器、交给 kopf 运行。
测试重点是"**参数与调用是否正确**"，而不是把 kopf 真的跑起来
（那需要集群，属于端到端验证的范畴）。
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from aetherserve.operator import main as operator_main


@pytest.fixture
def captured_kopf(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """注入假 kopf，捕获 run 的调用参数。"""
    captured: dict[str, Any] = {}

    fake = types.ModuleType("kopf")

    def run(**kwargs: Any) -> None:
        captured["run_kwargs"] = kwargs

    fake.run = run  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "kopf", fake)

    registered: list[str] = []
    monkeypatch.setattr(
        "aetherserve.operator.handlers.register_handlers",
        lambda: registered.append("ok"),
    )
    captured["registered"] = registered
    return captured


class TestMainEntryPoint:
    def test_main_registers_handlers_and_runs_kopf(self, captured_kopf) -> None:
        rc = operator_main.main([])
        assert rc == 0
        assert captured_kopf["registered"] == ["ok"], "启动时必须注册 kopf 处理器"
        assert "run_kwargs" in captured_kopf, "必须把控制权交给 kopf.run"

    def test_run_is_not_clusterwide(self, captured_kopf) -> None:
        """不做 cluster-wide watch。

        影响面最小化：只监听被 RBAC 授予的命名空间，
        避免运维误把 Operator 装成全集群权限。
        """
        operator_main.main([])
        assert captured_kopf["run_kwargs"]["clusterwide"] is False

    def test_liveness_endpoint_configured(self, captured_kopf) -> None:
        """必须暴露存活端点，否则 Deployment 的 livenessProbe 会一直失败。"""
        operator_main.main([])
        endpoint = captured_kopf["run_kwargs"]["liveness_endpoint"]
        assert endpoint.startswith("http://")
        assert endpoint.endswith("/healthz")

    def test_verbose_flag_selects_debug_level(self, captured_kopf, monkeypatch) -> None:
        """--verbose 应把日志级别切到 DEBUG（排障时常用）。"""
        import logging

        levels: list[int] = []

        def fake_basic_config(**kwargs: Any) -> None:
            levels.append(kwargs.get("level", 0))

        monkeypatch.setattr(logging, "basicConfig", fake_basic_config)

        operator_main.main(["--verbose"])
        assert levels and levels[0] == logging.DEBUG

    def test_default_level_is_info(self, captured_kopf, monkeypatch) -> None:
        import logging

        levels: list[int] = []

        def fake_basic_config(**kwargs: Any) -> None:
            levels.append(kwargs.get("level", 0))

        monkeypatch.setattr(logging, "basicConfig", fake_basic_config)

        operator_main.main([])
        assert levels and levels[0] == logging.INFO

    def test_argv_none_uses_sys_argv(self, captured_kopf, monkeypatch) -> None:
        """不传 argv 时应回退到 sys.argv（真实运行时就是这条路径）。"""
        import logging

        levels: list[int] = []
        monkeypatch.setattr(logging, "basicConfig", lambda **kw: levels.append(kw.get("level", 0)))
        monkeypatch.setattr(sys, "argv", ["aetherserve-operator", "--verbose"])

        operator_main.main(None)
        assert levels and levels[0] == logging.DEBUG
