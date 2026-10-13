"""kopf 处理器注册与回调测试：用假 kopf 模块驱动真实处理器。

为什么值得测？
-------------
``register_handlers`` 里的三个回调（create/update/resume 共用的 ``reconcile``、
周期灰度分析 ``periodic_analysis``、以及生命周期钩子）此前完全没有覆盖。
它们承载了关键的运行时语义：

- **resume 必须注册**：没有它，Operator 重启后已存在的 CR 永远不会被重新调和；
- **调和失败必须抛 TemporaryError**：让 kopf 按退避重试，而不是静默失败；
- **灰度分析必须用 return 而不是 raise**：``kopf.Patch`` 是 dict 子类而非异常
  （这里曾有一个真实 bug：写成 raise 会让灰度分析每 60s 崩一次）。

用假 kopf 模块可以在毫秒级验证这些语义，无需真实集群。
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from aetherserve.operator import handlers

CR = {
    "apiVersion": "aetherserve.io/v1alpha1",
    "kind": "InferenceService",
    "metadata": {"name": "demo", "namespace": "ns", "uid": "uid-demo", "generation": 1},
}


class FakePatch(dict):
    """模仿 kopf.Patch：它是 dict 子类，**不是异常**。"""


class FakeTemporaryError(Exception):
    def __init__(self, message: str, delay: float = 0) -> None:
        super().__init__(message)
        self.delay = delay


class NullDecorator:
    """模拟 kopf 的装饰器：登记回调并原样返回函数。"""

    def __init__(self, registry: dict[str, Any], name: str) -> None:
        self.registry = registry
        self.name = name

    def __call__(self, *args: Any, **kwargs: Any):  # type: ignore[no-untyped-def]
        # 支持 @kopf.on.create(group, version, plural) 与 @kopf.on.startup()
        def decorator(fn):  # type: ignore[no-untyped-def]
            self.registry[self.name] = fn
            return fn

        if len(args) == 1 and callable(args[0]) and not kwargs:
            return decorator(args[0])
        return decorator


class DecoratorGroup:
    """模拟 kopf.on.* / kopf.timer / kopf.on.field 的命名空间。"""

    def __init__(self, registry: dict[str, Any], prefix: str) -> None:
        self.registry = registry
        self.prefix = prefix

    def __getattr__(self, item: str) -> NullDecorator:
        return NullDecorator(self.registry, f"{self.prefix}.{item}")


@pytest.fixture
def fake_kopf(monkeypatch: pytest.MonkeyPatch):
    """注入假 kopf 模块并注册处理器，返回回调注册表。"""
    registry: dict[str, Any] = {}

    fake = types.ModuleType("kopf")
    fake.Patch = FakePatch  # type: ignore[attr-defined]
    fake.TemporaryError = FakeTemporaryError  # type: ignore[attr-defined]
    fake.on = DecoratorGroup(registry, "on")  # type: ignore[attr-defined]
    fake.timer = lambda *a, **kw: NullDecorator(registry, "timer")  # type: ignore[attr-defined]
    # kopf.on.field 作为属性访问应返回装饰器工厂
    monkeypatch.setitem(sys.modules, "kopf", fake)

    handlers.register_handlers()
    return registry


class TestHandlerRegistration:
    def test_registers_all_required_callbacks(self, fake_kopf) -> None:
        """必须注册的回调集合。缺 resume 会导致 Operator 重启后不再接管已有 CR。"""
        required = {
            "on.startup",
            "on.cleanup",
            "on.create",
            "on.update",
            "on.resume",  # ★ 最易被漏掉
            "timer",
        }
        missing = required - set(fake_kopf)
        assert not missing, f"缺少必要的 kopf 回调注册: {missing}"

    def test_create_update_resume_share_one_function(self, fake_kopf) -> None:
        """三者共用同一调和函数是正确的：无论怎么被触发，都应收敛到期望状态。"""
        assert fake_kopf["on.create"] is fake_kopf["on.update"]
        assert fake_kopf["on.update"] is fake_kopf["on.resume"]


class TestReconcileCallback:
    async def test_reconcile_creates_resources_and_returns_status(
        self, fake_kopf, fake_k8s, monkeypatch
    ) -> None:
        """调和回调应执行计划并回传 status（供 kopf 写回 CR）。"""
        applied: list[str] = []

        def fake_apply(plan):  # type: ignore[no-untyped-def]
            applied.append("called")
            return {"Deployment/demo-gateway": "created"}

        monkeypatch.setattr(handlers, "_apply_plan", fake_apply)
        monkeypatch.setattr(handlers, "_list_owned", lambda ns, name, uid=None: [])
        monkeypatch.setattr(handlers, "_fetch_canary_metrics", lambda ns, name: handlers.CanaryMetrics())

        result = await fake_kopf["on.create"](
            spec={"model": "m", "replicas": 1},
            meta=CR["metadata"],
            body={"status": {}},
        )
        assert applied == ["called"]
        assert "status" in result
        assert result["status"]["observedGeneration"] == 1
        assert "applied" in result and "summary" in result

    async def test_reconcile_uses_owner_uid_when_listing(self, fake_kopf, fake_k8s, monkeypatch) -> None:
        """列举子资源时必须传入 owner uid，否则会把别人的资源当成自己的。"""
        captured: dict[str, Any] = {}

        def fake_list(ns: str, name: str, uid: str | None = None):  # type: ignore[no-untyped-def]
            captured["ns"] = ns
            captured["name"] = name
            captured["uid"] = uid
            return []

        monkeypatch.setattr(handlers, "_list_owned", fake_list)
        monkeypatch.setattr(handlers, "_apply_plan", lambda plan: {})
        monkeypatch.setattr(handlers, "_fetch_canary_metrics", lambda ns, name: handlers.CanaryMetrics())

        await fake_kopf["on.create"](spec={"model": "m"}, meta=CR["metadata"], body={})
        assert captured["uid"] == "uid-demo"
        assert captured["ns"] == "ns"

    async def test_reconcile_raises_temporary_error_on_failure(
        self, fake_kopf, fake_k8s, monkeypatch
    ) -> None:
        """调和失败必须抛 TemporaryError，让 kopf 按退避重试 —— 不能静默吞掉。"""
        monkeypatch.setattr(handlers, "_list_owned", lambda ns, name, uid=None: [])
        monkeypatch.setattr(handlers, "_apply_plan", lambda plan: {})
        monkeypatch.setattr(handlers, "_fetch_canary_metrics", lambda ns, name: handlers.CanaryMetrics())

        def boom(plan):  # type: ignore[no-untyped-def]
            raise RuntimeError("模拟写集群失败")

        # 让 plan_reconcile 内部抛错
        monkeypatch.setattr(handlers, "plan_reconcile", boom)

        with pytest.raises(FakeTemporaryError) as ei:
            await fake_kopf["on.create"](spec={"model": "m"}, meta=CR["metadata"], body={})
        assert "调和失败" in str(ei.value) or "重试" in str(ei.value)

    async def test_reconcile_skips_apply_when_plan_has_errors(self, fake_kopf, fake_k8s, monkeypatch) -> None:
        """校验失败（plan.errors 非空）时不应尝试写集群。"""
        called: list[int] = []

        def fake_apply(plan):  # type: ignore[no-untyped-def]
            called.append(1)
            return {}

        monkeypatch.setattr(handlers, "_apply_plan", fake_apply)
        monkeypatch.setattr(handlers, "_list_owned", lambda ns, name, uid=None: [])
        monkeypatch.setattr(handlers, "_fetch_canary_metrics", lambda ns, name: handlers.CanaryMetrics())

        # echo 后端 + 申请 GPU 会触发校验错误
        result = await fake_kopf["on.create"](
            spec={"model": "m", "backend": "echo", "resources": {"gpu": 1}},
            meta=CR["metadata"],
            body={},
        )
        assert called == [], "校验失败却仍然写了集群"
        assert result["applied"] == {}


class TestPeriodicAnalysisCallback:
    async def test_returns_none_when_canary_disabled(self, fake_kopf, fake_k8s, monkeypatch) -> None:
        called: list[int] = []
        monkeypatch.setattr(handlers, "_apply_plan", lambda plan: called.append(1) or {})

        result = await fake_kopf["timer"](
            spec={"model": "m"},  # 未启用 canary
            meta=CR["metadata"],
            body={},
        )
        assert result is None
        assert called == [], "未启用灰度时不应执行任何计划"

    async def test_returns_patch_when_canary_enabled(self, fake_kopf, fake_k8s, monkeypatch) -> None:
        """灰度启用时必须 **return** patch 对象。

        这里曾有一个真实 bug：写成 ``raise kopf.Patch(...)``。
        而 ``kopf.Patch`` 是 ``dict`` 子类、不是异常，
        运行时会抛 "exceptions must derive from BaseException"，
        导致灰度分析每 60s 崩一次。本用例就是它的回归保护。
        """
        monkeypatch.setattr(handlers, "_list_owned", lambda ns, name, uid=None: [])
        monkeypatch.setattr(
            handlers, "_fetch_canary_metrics", lambda ns, name: handlers.CanaryMetrics(request_count=0)
        )

        result = await fake_kopf["timer"](
            spec={
                "model": "m",
                "replicas": 2,
                "canary": {"enabled": True, "initialWeight": 10, "minSamples": 10},
            },
            meta=CR["metadata"],
            body={"status": {"canaryState": "None", "canaryWeight": 0}},
        )
        assert isinstance(result, dict), "必须 return patch 对象而不是 raise"
        assert "status" in result
        assert result["status"]["canaryWeight"] == 10
        assert result["status"]["canaryState"] == "Progressing"

    async def test_canary_does_not_advance_without_metrics(self, fake_kopf, fake_k8s, monkeypatch) -> None:
        """拿不到 Prometheus 指标时灰度必须停在原地（fail-safe）。"""
        monkeypatch.setattr(handlers, "_list_owned", lambda ns, name, uid=None: [])
        monkeypatch.setattr(handlers, "_fetch_canary_metrics", lambda ns, name: handlers.CanaryMetrics())

        result = await fake_kopf["timer"](
            spec={
                "model": "m",
                "replicas": 2,
                "canary": {"enabled": True, "initialWeight": 30, "minSamples": 100},
            },
            meta=CR["metadata"],
            body={"status": {"canaryState": "Progressing", "canaryWeight": 30}},
        )
        assert result is not None
        assert result["status"]["canaryWeight"] == 30, "缺指标时权重应保持不变"


class TestLifecycleCallbacks:
    async def test_startup_and_cleanup_are_awaitable(self, fake_kopf) -> None:
        await fake_kopf["on.startup"]()
        await fake_kopf["on.cleanup"]()


class TestApiForCoverage:
    def test_all_built_kinds_route_to_a_known_group(self) -> None:
        """builders 产出的每种 Kind 都必须被 _api_for 正确路由。"""
        from aetherserve.operator import builders as b
        from aetherserve.operator.crd import parse_spec

        spec = parse_spec({"model": "m", "replicas": 3})
        kinds = {m["kind"] for m in b.build_all("demo", "ns", spec, CR, include_canary=True)}
        for kind in kinds:
            assert handlers._api_for(kind) in {"apps", "core", "autoscaling", "policy", "custom"}
