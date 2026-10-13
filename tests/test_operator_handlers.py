"""集群 I/O 层测试：用假 Kubernetes 客户端覆盖 ``operator/handlers.py``。

为什么必须测这一层？
-------------------
``handlers.py`` 是"读集群 → 调计划 → 写集群"的胶水层，此前完全没有测试覆盖。
常见误解是"它只能靠真实集群验证" —— 实际上它的分支相当明确：

- 按 Kind 选择 API 组（apps / core / autoscaling / policy / 动态客户端）
- 已存在（409）走 patch，其它错误走失败上报
- 冲突时 Service 必须剔除不可变字段
- Prometheus 不可用时要 fail-safe

这些都能用**假 API 客户端**精确验证，且毫秒级完成。
真正只能在集群里验证的只有"API Server 是否接受这份清单"，
那部分由 ``deploy/helm/verify_chart.py`` 的官方模型反序列化 + kind 端到端补齐。

测试手法说明
-----------
``handlers`` 内部用的是函数级延迟导入（``from kubernetes import client``），
因此不能用 ``monkeypatch.setattr(module, "client", ...)`` 直接替换模块属性。
共享夹具 ``fake_k8s``（定义在 ``conftest.py``）通过替换 ``sys.modules``
中的 ``kubernetes.client`` 来注入假模块 —— 效果与真实导入一致，
且由 ``monkeypatch`` 自动还原，不泄漏到其它测试。
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from aetherserve.operator import builders as b
from aetherserve.operator import handlers
from aetherserve.operator.crd import parse_spec
from aetherserve.operator.reconciler import ReconcileAction, ReconcilePlan

# 从 conftest 复用假客户端的异常类型（pytest 会把 tests/ 加入 sys.path）
from conftest import FakeApiException

CR = {
    "apiVersion": "aetherserve.io/v1alpha1",
    "kind": "InferenceService",
    "metadata": {"name": "demo", "namespace": "ns", "uid": "uid-demo", "generation": 1},
}


# ===================================================================== 假客户端


def deployment_manifest(name: str = "demo") -> dict[str, Any]:
    """构造与真实调和产物一致的 Deployment 清单。

    注意资源名是 ``<CR 名>-gateway``，不是模型名 —— 测试断言必须用真实名称，
    否则会出现"断言用名字 A、实际资源叫名字 B"的假失败。
    """
    spec = parse_spec({"model": "m", "replicas": 2})
    return b.build_deployment(name, "ns", spec, CR)


def service_manifest(name: str = "demo") -> dict[str, Any]:
    spec = parse_spec({"model": "m"})
    return b.build_service(name, "ns", spec, CR)


def identity_of(manifest: dict[str, Any]) -> str:
    """返回 ``Kind/name``，用于拼接 _apply_plan 的结果键。"""
    return f"{manifest['kind']}/{manifest['metadata']['name']}"


# ===================================================================== 路由
class TestApiRouting:
    @pytest.mark.parametrize(
        "kind,expected",
        [
            ("Deployment", "apps"),
            ("Service", "core"),
            ("HorizontalPodAutoscaler", "autoscaling"),
            ("PodDisruptionBudget", "policy"),
            ("ServiceMonitor", "custom"),
            ("PrometheusRule", "custom"),
        ],
    )
    def test_kind_maps_to_api_group(self, kind: str, expected: str) -> None:
        assert handlers._api_for(kind) == expected

    def test_unknown_kind_falls_back_to_custom(self) -> None:
        """未知 Kind 走动态客户端（CRD 类型），而不是静默丢弃。"""
        assert handlers._api_for("SomeCustomThing") == "custom"


# ===================================================================== 创建 / 冲突
class TestCreateOrPatch:
    def test_deployment_created(self, fake_k8s) -> None:
        recorder, _ = fake_k8s
        result = handlers._create_or_patch(deployment_manifest())
        assert result == "created"
        assert "create_namespaced_deployment" in recorder.methods()
        ns, _body = recorder.find("create_namespaced_deployment")  # type: ignore[misc]
        assert ns == "ns"

    def test_deployment_conflict_becomes_patch(self, fake_k8s) -> None:
        """已存在（409）必须退化为 patch —— 这是调和幂等的关键。"""
        recorder, state = fake_k8s
        state["behavior"] = "conflict"
        result = handlers._create_or_patch(deployment_manifest())
        assert result == "patched"
        assert "patch_namespaced_deployment" in recorder.methods()

    def test_service_conflict_patches_without_status(self, fake_k8s) -> None:
        """Service 的 clusterIP 等字段不可变，patch 时必须剔除 status。"""
        recorder, state = fake_k8s
        state["behavior"] = "conflict"
        manifest = service_manifest()
        manifest["status"] = {"loadBalancer": {"ingress": [{"ip": "1.2.3.4"}]}}

        result = handlers._create_or_patch(manifest)
        assert result == "patched"
        args = recorder.find("patch_namespaced_service")
        assert args is not None
        payload = args[2]
        assert "status" not in payload, "patch 携带 status 会触发不可变字段错误"

    def test_non_409_error_is_reported_not_swallowed(self, fake_k8s) -> None:
        """5xx 等错误不能当成"已存在"处理，必须计入失败指标并返回 skipped。"""
        recorder, state = fake_k8s
        state["behavior"] = "error"
        result = handlers._create_or_patch(deployment_manifest())
        assert result == "skipped"
        assert "patch_namespaced_deployment" not in recorder.methods(), "不应在非冲突错误时尝试 patch"

    def test_hpa_created(self, fake_k8s) -> None:
        recorder, _ = fake_k8s
        spec = parse_spec({"model": "m", "autoscaling": {"enabled": True}})
        manifest = b.build_hpa("demo", "ns", spec, CR)
        assert handlers._create_or_patch(manifest) == "created"
        assert "create_namespaced_horizontal_pod_autoscaler" in recorder.methods()

    def test_pdb_created(self, fake_k8s) -> None:
        recorder, _ = fake_k8s
        spec = parse_spec({"model": "m", "replicas": 3})
        manifest = b.build_pdb("demo", "ns", spec, CR)
        assert handlers._create_or_patch(manifest) == "created"
        assert "create_namespaced_pod_disruption_budget" in recorder.methods()

    def test_custom_resource_uses_dynamic_client(self, fake_k8s) -> None:
        """ServiceMonitor 这类 CRD 类型必须走动态客户端。"""
        recorder, _ = fake_k8s
        spec = parse_spec({"model": "m"})
        manifest = b.build_service_monitor("demo", "ns", spec, CR)
        assert handlers._create_or_patch(manifest) == "created"
        assert "resources_get" in recorder.methods()
        assert "dynamic_create" in recorder.methods()
        api_version, kind = recorder.find("resources_get")  # type: ignore[misc]
        assert kind == "ServiceMonitor"
        assert api_version.startswith("monitoring.coreos.com")

    def test_custom_resource_conflict_uses_merge_patch(self, fake_k8s) -> None:
        recorder, state = fake_k8s
        state["behavior"] = "conflict"
        spec = parse_spec({"model": "m"})
        manifest = b.build_prometheus_rule("demo", "ns", spec, CR)
        assert handlers._create_or_patch(manifest) == "patched"
        args = recorder.find("dynamic_patch")
        assert args is not None
        _body, name, ns = args[0], args[1], args[2]
        assert name == "demo-rules"
        assert ns == "ns"


# ===================================================================== 删除
class TestDeleteResource:
    @pytest.mark.parametrize(
        "kind,method",
        [
            ("Deployment", "delete_namespaced_deployment"),
            ("Service", "delete_namespaced_service"),
            ("HorizontalPodAutoscaler", "delete_namespaced_horizontal_pod_autoscaler"),
            ("PodDisruptionBudget", "delete_namespaced_pod_disruption_budget"),
        ],
    )
    def test_deletes_by_kind(self, fake_k8s, kind: str, method: str) -> None:
        recorder, _ = fake_k8s
        resource = {"kind": kind, "metadata": {"name": "demo-x", "namespace": "ns"}}
        assert handlers._delete_resource(resource) is True
        assert method in recorder.methods()

    def test_unknown_kind_returns_false(self, fake_k8s) -> None:
        assert handlers._delete_resource({"kind": "Weird", "metadata": {"name": "x"}}) is False

    def test_delete_notfound_does_not_raise(self, fake_k8s, monkeypatch) -> None:
        """404 表示目标已不存在 —— 对"确保删除"的语义来说不应抛异常。"""
        recorder, _ = fake_k8s

        class NotFoundApps:
            def delete_namespaced_deployment(self, name: str, ns: str) -> Any:
                recorder.add("delete_namespaced_deployment", (name, ns), {})
                raise FakeApiException(404, "NotFound")

        import kubernetes.client as kc

        monkeypatch.setattr(kc, "AppsV1Api", NotFoundApps)
        resource = {"kind": "Deployment", "metadata": {"name": "demo-x", "namespace": "ns"}}
        assert handlers._delete_resource(resource) is False


# ===================================================================== 列举与归属
class TestListOwned:
    def test_returns_empty_when_no_resources(self, fake_k8s) -> None:
        assert handlers._list_owned("ns", "demo", "uid-demo") == []

    def test_uses_managed_by_label_selector(self, fake_k8s) -> None:
        """必须按 managed-by 标签过滤 —— 否则会把别人的资源也列进来。"""
        recorder, _ = fake_k8s
        handlers._list_owned("ns", "demo", "uid-demo")
        for method in (
            "list_namespaced_deployment",
            "list_namespaced_service",
            "list_namespaced_horizontal_pod_autoscaler",
            "list_namespaced_pod_disruption_budget",
        ):
            assert method in recorder.methods(), f"未列出 {method}，会漏掉该类型的孤儿资源"
        _ns, kwargs = next((c[1], c[2]) for c in recorder.calls if c[0] == "list_namespaced_deployment")
        assert kwargs.get("label_selector") == f"{b.LABEL_MANAGED_BY}={b.MANAGED_BY}"

    def test_owned_by_uid_filters_foreign_resources(self, fake_k8s, monkeypatch) -> None:
        """归属判断用 uid：其它 CR 的资源不能被当成自己的（否则会误删）。"""

        class OneDeployment:
            def list_namespaced_deployment(self, ns: str, **kw: Any) -> Any:
                mine = types.SimpleNamespace(
                    metadata=types.SimpleNamespace(
                        name="demo-gateway",
                        ownerReferences=[types.SimpleNamespace(uid="uid-demo", kind="InferenceService")],
                    )
                )
                theirs = types.SimpleNamespace(
                    metadata=types.SimpleNamespace(
                        name="other-gateway",
                        ownerReferences=[types.SimpleNamespace(uid="uid-other", kind="InferenceService")],
                    )
                )
                return types.SimpleNamespace(items=[mine, theirs])

        import kubernetes.client as kc

        monkeypatch.setattr(kc, "AppsV1Api", OneDeployment)
        found = handlers._list_owned("ns", "demo", "uid-demo")
        assert [f["metadata"]["name"] for f in found] == ["demo-gateway"], (
            "应只保留 ownerReference.uid 匹配的资源，否则会误删其它 CR 的子资源"
        )

    def test_unknown_uid_falls_back_to_managed_by_label(self, fake_k8s, monkeypatch) -> None:
        """拿不到 uid 时（离线/单测）退化为按 managed-by 标签归属，保证逻辑可用。"""

        class OneDeployment:
            def list_namespaced_deployment(self, ns: str, **kw: Any) -> Any:
                item = types.SimpleNamespace(
                    metadata=types.SimpleNamespace(
                        name="demo-gateway",
                        ownerReferences=[],
                        # fallback 路径依赖 managed-by 标签，假数据必须带上，
                        # 否则测的不是代码逻辑而是"假数据不全"
                        labels={b.LABEL_MANAGED_BY: b.MANAGED_BY},
                    )
                )
                return types.SimpleNamespace(items=[item])

        import kubernetes.client as kc

        monkeypatch.setattr(kc, "AppsV1Api", OneDeployment)
        found = handlers._list_owned("ns", "demo", None)
        assert [f["metadata"]["name"] for f in found] == ["demo-gateway"]


# ===================================================================== Prometheus
class TestFetchCanaryMetrics:
    def test_no_url_returns_empty_metrics(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """未配置 Prometheus 时返回空指标 → 灰度会停在原地（fail-safe）。"""
        monkeypatch.delenv("AETHERSERVE_PROMETHEUS_URL", raising=False)
        m = handlers._fetch_canary_metrics("ns", "demo")
        assert m.request_count == 0
        assert m.error_count == 0

    def test_query_failure_returns_empty_metrics(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Prometheus 挂了也必须 fail-safe，而不是盲目推进灰度。"""
        monkeypatch.setenv("AETHERSERVE_PROMETHEUS_URL", "http://prometheus.invalid:9090")
        m = handlers._fetch_canary_metrics("ns", "demo")
        assert m.request_count == 0

    def test_parses_prometheus_response(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """正常响应要能解析出请求数与错误数。"""
        monkeypatch.setenv("AETHERSERVE_PROMETHEUS_URL", "http://prom:9090")

        responses = {
            "total": {"data": {"result": [{"value": [0, "1000"]}]}},
            "error": {"data": {"result": [{"value": [0, "25"]}]}},
        }

        class FakeResponse:
            def __init__(self, payload: dict) -> None:
                self._payload = payload

            def json(self) -> dict:
                return self._payload

        class FakeHttpxClient:
            def __init__(self, *a: Any, **kw: Any) -> None:
                pass

            def __enter__(self) -> FakeHttpxClient:
                return self

            def __exit__(self, *a: Any) -> None:
                return None

            def get(self, url: str, params: dict) -> FakeResponse:
                is_error = "error" in params["query"]
                return FakeResponse(responses["error" if is_error else "total"])

        fake_httpx = types.ModuleType("httpx")
        fake_httpx.Client = FakeHttpxClient  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "httpx", fake_httpx)

        m = handlers._fetch_canary_metrics("ns", "demo")
        assert m.request_count == 1000
        assert m.error_count == 25
        assert m.error_rate == pytest.approx(0.025)

    def test_empty_result_means_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AETHERSERVE_PROMETHEUS_URL", "http://prom:9090")

        class FakeResponse:
            def json(self) -> dict:
                return {"data": {"result": []}}

        class FakeHttpxClient:
            def __init__(self, *a: Any, **kw: Any) -> None:
                pass

            def __enter__(self) -> FakeHttpxClient:
                return self

            def __exit__(self, *a: Any) -> None:
                return None

            def get(self, url: str, params: dict) -> FakeResponse:
                return FakeResponse()

        fake_httpx = types.ModuleType("httpx")
        fake_httpx.Client = FakeHttpxClient  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "httpx", fake_httpx)

        m = handlers._fetch_canary_metrics("ns", "demo")
        assert m.request_count == 0


# ===================================================================== 计划执行
class TestApplyPlan:
    def test_apply_plan_dispatches_by_op(self, fake_k8s) -> None:
        recorder, _ = fake_k8s
        dep_new = deployment_manifest()
        svc_upd = service_manifest()
        dep_noop = deployment_manifest("demo-noop")
        plan = ReconcilePlan(
            actions=[
                ReconcileAction("create", dep_new, reason="new"),
                ReconcileAction("update", svc_upd, reason="changed"),
                ReconcileAction("noop", dep_noop, reason="same"),
                ReconcileAction(
                    "delete",
                    {"kind": "Service", "metadata": {"name": "demo-canary", "namespace": "ns"}},
                    reason="orphan",
                ),
            ]
        )
        results = handlers._apply_plan(plan)
        assert results[identity_of(dep_new)] == "created"
        assert results[identity_of(svc_upd)] == "created"
        assert results[identity_of(dep_noop)] == "noop"
        assert results["Service/demo-canary"] == "deleted"
        assert "create_namespaced_service" in recorder.methods()
        assert "delete_namespaced_service" in recorder.methods()
        # noop 不应产生任何写操作
        create_calls = [m for m in recorder.methods() if m.startswith("create_")]
        assert len(create_calls) == 2, f"noop 动作产生了多余的写操作: {create_calls}"

    def test_apply_empty_plan(self, fake_k8s) -> None:
        assert handlers._apply_plan(ReconcilePlan()) == {}

    def test_apply_plan_reports_skipped_on_failure(self, fake_k8s) -> None:
        """写集群失败要如实记录为 skipped，不能假装成功。"""
        _recorder, state = fake_k8s
        state["behavior"] = "error"
        manifest = deployment_manifest()
        plan = ReconcilePlan(actions=[ReconcileAction("create", manifest, reason="new")])
        results = handlers._apply_plan(plan)
        assert results[identity_of(manifest)] == "skipped"


# ===================================================================== 反序列化
class TestDeserialize:
    def test_deserialize_accepts_manifest_dict(self, fake_k8s) -> None:
        obj = handlers._deserialize({"kind": "Deployment"}, "V1Deployment")
        assert obj.kind == "Deployment"

    def test_api_for_covers_all_builder_kinds(self) -> None:
        """builders 生成的所有 Kind 都必须有路由，否则会走错 API 组。"""
        spec = parse_spec({"model": "m", "replicas": 3})
        kinds = {m["kind"] for m in b.build_all("demo", "ns", spec, CR, include_canary=True)}
        routed = {k: handlers._api_for(k) for k in kinds}
        assert all(v for v in routed.values()), f"有 Kind 未路由: {routed}"
        assert routed["Deployment"] == "apps"
        assert routed["Service"] == "core"
