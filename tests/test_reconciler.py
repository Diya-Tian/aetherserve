"""调和逻辑测试：幂等性、变更检测、孤儿清理、状态回写。

这是 Operator 最该被测试的部分 —— 逻辑全在这里，且不需要集群。
"""

from __future__ import annotations

import json
from typing import Any

from aetherserve.operator import builders as b
from aetherserve.operator.canary import CanaryMetrics
from aetherserve.operator.crd import API_GROUP, API_VERSION, KIND
from aetherserve.operator.reconciler import plan_reconcile


def make_cr(spec: dict[str, Any] | None = None, status: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "apiVersion": f"{API_GROUP}/{API_VERSION}",
        "kind": KIND,
        "metadata": {
            "name": "qwen-demo",
            "namespace": "ai-serving",
            "uid": "uid-xyz-789",
            "generation": 3,
        },
        "spec": {"model": "Qwen/Qwen2.5-1.5B-Instruct", **(spec or {})},
        "status": status or {},
    }


def existing_from_plan(
    cr: dict[str, Any], spec: dict[str, Any] | None = None, include_canary: bool = False
) -> list[dict[str, Any]]:
    """用"上一轮调和的结果"模拟集群里已存在的资源（含 status）。"""
    from aetherserve.operator.crd import parse_spec

    parsed = parse_spec(cr["spec"])
    resources = b.build_all("qwen-demo", "ai-serving", parsed, cr, include_canary=include_canary)
    for r in resources:
        if r["kind"] == "Deployment":
            r["status"] = {"readyReplicas": parsed.replicas}
    return resources


class TestFirstReconcile:
    def test_all_resources_created(self) -> None:
        cr = make_cr({"replicas": 2})
        plan = plan_reconcile(cr, existing=[])
        ops = {a.op for a in plan.actions}
        assert ops == {"create"}, f"首次调和应全部创建，实际 {[str(a) for a in plan.actions]}"
        assert not plan.errors

    def test_creates_expected_resource_set(self) -> None:
        cr = make_cr({"replicas": 2})
        plan = plan_reconcile(cr, existing=[])
        kinds = [a.identity[1] for a in plan.actions]
        assert kinds == [
            "Deployment",
            "Service",
            "HorizontalPodAutoscaler",
            "ServiceMonitor",
            "PrometheusRule",
            "PodDisruptionBudget",
        ]

    def test_status_written(self) -> None:
        cr = make_cr({"replicas": 2})
        plan = plan_reconcile(cr, existing=[])
        assert plan.status.observed_generation == 3
        assert plan.status.endpoint.startswith("http://qwen-demo-stable.ai-serving.svc")
        ready = plan.status.get_condition("Ready")
        assert ready is not None
        # 资源还没就绪 → Ready=False，但 Progressing=True
        assert ready.status == "False"
        progressing = plan.status.get_condition("Progressing")
        assert progressing is not None and progressing.status == "True"


class TestIdempotency:
    def test_second_reconcile_is_noop(self) -> None:
        """核心不变量：第二次调和不得产生任何写操作（否则会无限重启 Pod）。"""
        cr = make_cr({"replicas": 2})
        first = plan_reconcile(cr, existing=[])
        # 模拟第一轮已 apply：把期望资源当成已存在资源
        applied = [a.manifest for a in first.actions if a.op == "create"]
        for r in applied:
            if r["kind"] == "Deployment":
                r["status"] = {"readyReplicas": 2}

        second = plan_reconcile(cr, existing=applied)
        writes = [str(a) for a in second.actions if a.op in ("create", "update", "delete")]
        assert writes == [], f"第二次调和产生了写操作：{writes}"

    def test_repeated_reconcile_stable(self) -> None:
        cr = make_cr({"replicas": 2})
        baseline = json.dumps(
            [b.manifest_identity(a.manifest) for a in plan_reconcile(cr, []).actions], sort_keys=True
        )
        for _ in range(5):
            current = json.dumps(
                [b.manifest_identity(a.manifest) for a in plan_reconcile(cr, []).actions], sort_keys=True
            )
            assert current == baseline

    def test_ready_status_after_pods_ready(self) -> None:
        cr = make_cr({"replicas": 2})
        existing = existing_from_plan(cr)
        plan = plan_reconcile(cr, existing=existing)
        assert plan.status.ready_replicas == 2
        ready = plan.status.get_condition("Ready")
        assert ready is not None and ready.status == "True"


class TestChangeDetection:
    def test_spec_change_triggers_update(self) -> None:
        cr = make_cr({"replicas": 2, "maxBatchSize": 8})
        existing = existing_from_plan(cr)

        # 修改批大小
        cr2 = make_cr({"replicas": 2, "maxBatchSize": 32})
        plan = plan_reconcile(cr2, existing=existing)
        updates = [a for a in plan.actions if a.op == "update" and a.identity[1] == "Deployment"]
        assert len(updates) == 1, "spec 变化必须触发 Deployment 更新（否则配置不生效）"
        assert "spec hash" in updates[0].reason

    def test_replica_change_triggers_update(self) -> None:
        cr = make_cr({"replicas": 1})
        existing = existing_from_plan(cr)
        cr2 = make_cr({"replicas": 5})
        plan = plan_reconcile(cr2, existing=existing)
        dep_update = [a for a in plan.actions if a.op == "update" and a.identity[1] == "Deployment"]
        assert len(dep_update) == 1

    def test_unrelated_change_does_not_update(self) -> None:
        """只有灰度状态变化（spec 不变）时，不应重新写 Deployment。"""
        cr = make_cr({"replicas": 2})
        existing = existing_from_plan(cr)
        plan = plan_reconcile(cr, existing=existing)
        assert not [a for a in plan.actions if a.op in ("create", "update")]


class TestOrphanCleanup:
    def test_canary_resources_deleted_on_rollback(self) -> None:
        """灰度回滚后，canary 的 Deployment/Service 必须被清理，否则会一直占着 GPU。"""
        spec = {"replicas": 2, "canary": {"enabled": True, "initialWeight": 10, "step": 20, "minSamples": 10}}
        cr = make_cr(spec, status={"canaryState": "Progressing", "canaryWeight": 30})
        existing = existing_from_plan(cr, include_canary=True)
        # 确认存在 canary 资源
        assert any("canary" in r["metadata"]["name"] for r in existing)

        # 触发回滚：高错误率
        plan = plan_reconcile(cr, existing, canary_metrics=CanaryMetrics(request_count=1000, error_count=500))
        deletes = [a for a in plan.actions if a.op == "delete"]
        deleted_names = {a.identity[2] for a in deletes}
        assert "qwen-demo-canary" in deleted_names
        assert plan.status.canary_state == "RolledBack"
        assert plan.status.canary_weight == 0

    def test_orphan_cleanup_requires_managed_by_label(self) -> None:
        """安全检查：不带 managed-by 标签的资源绝不能被删除（避免误删他人资源）。"""
        cr = make_cr({"replicas": 1, "autoscaling": {"enabled": False}, "serviceMonitor": False})
        foreign = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {"name": "someone-elses-deployment", "namespace": "ai-serving", "labels": {}},
            "spec": {},
        }
        plan = plan_reconcile(cr, existing=[foreign])
        deletes = [a for a in plan.actions if a.op == "delete"]
        assert deletes == [], "不应删除非本 Operator 管理的资源"

    def test_autoscaling_disabled_removes_hpa(self) -> None:
        cr_enabled = make_cr({"replicas": 2})
        existing = existing_from_plan(cr_enabled)
        assert any(r["kind"] == "HorizontalPodAutoscaler" for r in existing)

        cr_disabled = make_cr({"replicas": 2, "autoscaling": {"enabled": False}})
        plan = plan_reconcile(cr_disabled, existing=existing)
        deletes = [a for a in plan.actions if a.op == "delete"]
        assert any(a.identity[1] == "HorizontalPodAutoscaler" for a in deletes), (
            "关闭自动扩缩容后 HPA 必须被删除"
        )


class TestValidationErrors:
    def test_invalid_spec_blocks_reconcile(self) -> None:
        cr = make_cr({"backend": "echo", "resources": {"gpu": 1}})
        plan = plan_reconcile(cr, existing=[])
        assert plan.errors
        assert plan.actions == [], "校验失败时不应创建任何资源"

    def test_missing_name(self) -> None:
        cr = {"metadata": {"namespace": "ns"}, "spec": {"model": "m"}}
        plan = plan_reconcile(cr, existing=[])
        assert any("name" in e for e in plan.errors)

    def test_empty_model(self) -> None:
        cr = make_cr({"model": ""})
        plan = plan_reconcile(cr, existing=[])
        assert plan.errors


class TestCanaryIntegration:
    def test_canary_creates_extra_resources(self) -> None:
        spec = {"replicas": 2, "canary": {"enabled": True, "initialWeight": 10, "minSamples": 10}}
        cr = make_cr(spec, status={"canaryState": "None", "canaryWeight": 0})
        plan = plan_reconcile(cr, existing=[])
        names = {a.identity[2] for a in plan.actions}
        assert "qwen-demo-canary" in names, "灰度启用时应创建 canary Deployment"
        assert plan.status.canary_state == "Progressing"
        assert plan.status.canary_weight == 10

    def test_canary_progresses_on_healthy_metrics(self) -> None:
        spec = {"replicas": 2, "canary": {"enabled": True, "initialWeight": 10, "step": 20, "minSamples": 10}}
        cr = make_cr(spec, status={"canaryState": "Progressing", "canaryWeight": 10})
        plan = plan_reconcile(cr, [], canary_metrics=CanaryMetrics(request_count=500, error_count=1))
        assert plan.status.canary_weight == 30
        assert plan.status.canary_state == "Progressing"
        assert plan.events and "canary" in plan.events[0]

    def test_canary_promotion_stops_canary_resources(self) -> None:
        spec = {"replicas": 2, "canary": {"enabled": True, "initialWeight": 90, "step": 20, "minSamples": 10}}
        cr = make_cr(spec, status={"canaryState": "Progressing", "canaryWeight": 90})
        plan = plan_reconcile(cr, [], canary_metrics=CanaryMetrics(request_count=500, error_count=0))
        assert plan.status.canary_state == "Promoted"
        assert plan.status.canary_weight == 100
        names = {a.identity[2] for a in plan.actions}
        assert "qwen-demo-canary" not in names, "全量后不再需要独立的 canary 资源"

    def test_canary_records_failure_condition(self) -> None:
        spec = {
            "replicas": 2,
            "canary": {"enabled": True, "initialWeight": 30, "maxErrorRate": 0.01, "minSamples": 10},
        }
        cr = make_cr(spec, status={"canaryState": "Progressing", "canaryWeight": 30})
        plan = plan_reconcile(cr, [], canary_metrics=CanaryMetrics(request_count=1000, error_count=100))
        cond = plan.status.get_condition("CanaryFailed")
        assert cond is not None and cond.status == "True"
        assert cond.reason == "ErrorRateExceeded"
        # 回滚后稳定版仍应对外服务
        ready = plan.status.get_condition("Ready")
        assert ready is not None and ready.status == "True"

    def test_canary_metrics_missing_does_not_advance(self) -> None:
        """监控不可用时（拿不到指标）灰度必须停在原地，不能盲目推进。"""
        spec = {"replicas": 2, "canary": {"enabled": True, "initialWeight": 10, "step": 20, "minSamples": 10}}
        cr = make_cr(spec, status={"canaryState": "Progressing", "canaryWeight": 10})
        plan = plan_reconcile(cr, [], canary_metrics=CanaryMetrics())
        assert plan.status.canary_weight == 10


class TestSummary:
    def test_summary_shape(self) -> None:
        cr = make_cr({"replicas": 2})
        plan = plan_reconcile(cr, existing=[])
        s = plan.summary()
        assert "actions" in s
        assert s["ready"] in (True, False)
        assert "endpoint" in s


class TestResourceDeterminism:
    def test_plan_manifests_match_builders(self) -> None:
        """调和产生的清单必须与 builders 直接生成的完全一致（防止逻辑分叉）。"""
        from aetherserve.operator.crd import parse_spec

        cr = make_cr({"replicas": 2})
        plan = plan_reconcile(cr, existing=[])
        direct = b.build_all("qwen-demo", "ai-serving", parse_spec(cr["spec"]), cr)
        assert [json.dumps(a.manifest, sort_keys=True) for a in plan.actions if a.op == "create"] == [
            json.dumps(m, sort_keys=True) for m in direct
        ]

    def test_no_canary_when_disabled(self) -> None:
        cr = make_cr({"replicas": 2})
        plan = plan_reconcile(cr, existing=[])
        assert all("canary" not in a.identity[2] for a in plan.actions)


class TestGenerationTracking:
    def test_observed_generation_recorded(self) -> None:
        cr = make_cr({"replicas": 1})
        cr["metadata"]["generation"] = 42
        plan = plan_reconcile(cr, existing=[])
        assert plan.status.observed_generation == 42

    def test_missing_generation_defaults_zero(self) -> None:
        cr = make_cr({"replicas": 1})
        cr["metadata"].pop("generation")
        plan = plan_reconcile(cr, existing=[])
        assert plan.status.observed_generation == 0
