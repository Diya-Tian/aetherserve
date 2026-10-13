"""Operator 测试：CRD 解析/校验/CRD schema/清单生成。

**重点：不依赖真实集群。**

清单合法性用 ``kubernetes`` 官方客户端的反序列化器验证 ——
把生成的 dict 解析成 ``V1Deployment`` / ``V1Service`` / ``V1HorizontalPodAutoscaler``
等类型化模型。这比"能 dump 成 YAML"强得多：字段拼错、类型不对都会直接报错。
"""

from __future__ import annotations

import json
from typing import Any

from kubernetes import client as k8s_client

from aetherserve.operator import builders as b
from aetherserve.operator.crd import (
    API_GROUP,
    API_VERSION,
    KIND,
    InferenceServiceSpec,
    crd_manifest,
    parse_spec,
    spec_hash,
    validate_spec,
)

API = k8s_client.ApiClient()

CR = {
    "apiVersion": f"{API_GROUP}/{API_VERSION}",
    "kind": KIND,
    "metadata": {"name": "qwen-demo", "namespace": "ai-serving", "uid": "uid-abc-123", "generation": 1},
}


def deserialize(manifest: dict[str, Any], model: str) -> Any:
    return API._ApiClient__deserialize(manifest, model)


# ===================================================================== CRD 模型
class TestParseSpec:
    def test_minimal_spec(self) -> None:
        spec = parse_spec({"model": "meta-llama/Llama-3-8B"})
        assert spec.model == "meta-llama/Llama-3-8B"
        assert spec.backend == "vllm"
        assert spec.replicas == 1
        assert spec.autoscaling.enabled is True

    def test_camel_case_fields(self) -> None:
        spec = parse_spec(
            {
                "model": "m",
                "maxModelLen": 8192,
                "gpuMemoryUtilization": 0.9,
                "maxBatchSize": 32,
                "maxWaitMs": 15.5,
                "serviceMonitor": False,
                "nodeSelector": {"zone": "gpu-a"},
            }
        )
        assert spec.max_model_len == 8192
        assert spec.gpu_memory_utilization == 0.9
        assert spec.max_batch_size == 32
        assert spec.max_wait_ms == 15.5
        assert spec.service_monitor is False
        assert spec.node_selector == {"zone": "gpu-a"}

    def test_snake_case_also_accepted(self) -> None:
        """宽容解析：CRD 演进时旧 CR 仍可被处理。"""
        spec = parse_spec({"model": "m", "max_model_len": 2048, "max_batch_size": 8})
        assert spec.max_model_len == 2048
        assert spec.max_batch_size == 8

    def test_nested_resources_and_autoscaling(self) -> None:
        spec = parse_spec(
            {
                "model": "m",
                "resources": {"gpu": 2, "cpuRequest": "2", "memoryLimit": "16Gi"},
                "autoscaling": {
                    "minReplicas": 2,
                    "maxReplicas": 10,
                    "targetQueueDepth": 32,
                    "metrics": ["queue_depth", "kv_cache"],
                },
            }
        )
        assert spec.resources.gpu == 2
        assert spec.resources.cpu_request == "2"
        assert spec.resources.memory_limit == "16Gi"
        assert spec.autoscaling.max_replicas == 10
        assert spec.autoscaling.metrics == ["queue_depth", "kv_cache"]

    def test_unknown_fields_ignored(self) -> None:
        spec = parse_spec({"model": "m", "totallyUnknownField": {"a": 1}})
        assert spec.model == "m"

    def test_gateway_env_translation(self) -> None:
        """spec 必须能翻译成网关的环境变量（这是 CRD 与业务代码的契约）。"""
        spec = parse_spec(
            {
                "model": "Qwen/Qwen2.5-1.5B-Instruct",
                "backend": "vllm",
                "quantization": "awq",
                "maxBatchSize": 24,
                "maxWaitMs": 12.0,
                "env": {"CUSTOM": "1"},
            }
        )
        env = spec.gateway_env()
        assert env["AETHERSERVE_MODEL"] == "Qwen/Qwen2.5-1.5B-Instruct"
        assert env["AETHERSERVE_BACKEND"] == "vllm"
        assert env["AETHERSERVE_QUANTIZATION"] == "awq"
        assert env["AETHERSERVE_BATCH_MAX_BATCH_SIZE"] == "24"
        assert env["AETHERSERVE_BATCH_MAX_WAIT_MS"] == "12.0"
        assert env["CUSTOM"] == "1"


class TestValidation:
    def test_valid_spec(self) -> None:
        assert validate_spec(parse_spec({"model": "m", "replicas": 2, "resources": {"gpu": 1}})) == []

    def test_empty_model(self) -> None:
        assert any("model" in e for e in validate_spec(InferenceServiceSpec(model="  ")))

    def test_negative_replicas(self) -> None:
        spec = InferenceServiceSpec(model="m", replicas=-1)
        assert any("replicas" in e for e in validate_spec(spec))

    def test_echo_backend_with_gpu_rejected(self) -> None:
        """echo 是假后端，申请 GPU 属配置错误（会白占卡）。"""
        spec = parse_spec({"model": "m", "backend": "echo", "resources": {"gpu": 1}})
        errors = validate_spec(spec)
        assert any("echo" in e for e in errors)

    def test_autoscaling_bounds(self) -> None:
        spec = parse_spec({"model": "m", "autoscaling": {"minReplicas": 5, "maxReplicas": 2}})
        assert any("maxReplicas" in e for e in validate_spec(spec))

    def test_max_replicas_below_replicas(self) -> None:
        spec = parse_spec({"model": "m", "replicas": 10, "autoscaling": {"minReplicas": 1, "maxReplicas": 3}})
        assert any("maxReplicas" in e for e in validate_spec(spec))

    def test_canary_without_replicas_warns(self) -> None:
        spec = parse_spec(
            {
                "model": "m",
                "replicas": 1,
                "canary": {"enabled": True},
                "autoscaling": {"enabled": False},
            }
        )
        assert any("灰度" in e for e in validate_spec(spec))

    def test_gpu_utilization_range(self) -> None:
        spec = InferenceServiceSpec(model="m", gpu_memory_utilization=1.5)
        assert any("gpuMemoryUtilization" in e for e in validate_spec(spec))


class TestSpecHash:
    def test_stable_across_key_order(self) -> None:
        """键顺序不同必须得到同一个哈希，否则会引发无意义的滚动重启。"""
        assert spec_hash({"a": 1, "b": {"c": 2}}) == spec_hash({"b": {"c": 2}, "a": 1})

    def test_changes_with_value(self) -> None:
        assert spec_hash({"a": 1}) != spec_hash({"a": 2})

    def test_spec_hash_method(self) -> None:
        s1 = parse_spec({"model": "m", "maxBatchSize": 8})
        s2 = parse_spec({"model": "m", "maxBatchSize": 8})
        s3 = parse_spec({"model": "m", "maxBatchSize": 16})
        assert s1.hash() == s2.hash()
        assert s1.hash() != s3.hash()


class TestCrdManifest:
    def test_shape(self) -> None:
        crd = crd_manifest()
        assert crd["kind"] == "CustomResourceDefinition"
        assert crd["metadata"]["name"] == "inferenceservices.aetherserve.io"
        spec = crd["spec"]
        assert spec["group"] == API_GROUP
        assert spec["scope"] == "Namespaced"
        assert spec["names"]["kind"] == KIND
        assert spec["names"]["shortNames"] == ["isvc"]

    def test_is_valid_yaml_serializable(self) -> None:
        json.dumps(crd_manifest())  # 必须可序列化（否则 kubectl apply 会失败）

    def test_printer_columns_present(self) -> None:
        """additionalPrinterColumns 决定 kubectl get 的显示，是可用性的一部分。"""
        cols = crd_manifest()["spec"]["versions"][0]["additionalPrinterColumns"]
        names = {c["name"] for c in cols}
        assert {"Model", "Backend", "Replicas", "Ready", "Canary"} <= names

    def test_schema_declares_required_model(self) -> None:
        schema = crd_manifest()["spec"]["versions"][0]["schema"]["openAPIV3Schema"]
        assert schema["properties"]["spec"]["required"] == ["model"]

    def test_backend_enum_matches_registry(self) -> None:
        """CRD 的 backend 枚举必须与代码里的后端注册表一致，否则用户会写出跑不起来的 CR。"""
        from aetherserve.backends import available_backends

        schema = crd_manifest()["spec"]["versions"][0]["schema"]["openAPIV3Schema"]
        declared = set(schema["properties"]["spec"]["properties"]["backend"]["enum"])
        assert declared == set(available_backends()), f"CRD 枚举 {declared} 与后端注册表不一致"


# ===================================================================== 清单构建
class TestLabels:
    def test_standard_labels(self) -> None:
        labels = b.labels_for("demo")
        assert labels[b.LABEL_APP] == "aetherserve"
        assert labels[b.LABEL_INSTANCE] == "demo"
        assert labels[b.LABEL_MANAGED_BY] == b.MANAGED_BY

    def test_selector_labels_are_stable(self) -> None:
        """selector 必须与 spec 无关（K8s 禁止修改 selector）。"""
        assert b.selector_labels("demo") == b.selector_labels("demo")
        s1 = parse_spec({"model": "a"})
        s2 = parse_spec({"model": "b", "replicas": 9})
        m1 = b.build_deployment("demo", "ns", s1, CR)
        m2 = b.build_deployment("demo", "ns", s2, CR)
        assert m1["spec"]["selector"] == m2["spec"]["selector"]

    def test_canary_selector_differs_from_stable(self) -> None:
        """稳定版与灰度的 selector 必须互斥，否则灰度流量会漏到旧版本。"""
        spec = parse_spec({"model": "m"})
        stable = b.build_deployment("demo", "ns", spec, CR)
        canary = b.build_deployment("demo", "ns", spec, CR, canary=True)
        assert stable["spec"]["selector"] != canary["spec"]["selector"]

        # 更强的断言：灰度的 selector 不能匹配稳定版的 Pod，反之亦然
        stable_pod_labels = stable["spec"]["template"]["metadata"]["labels"]
        canary_pod_labels = canary["spec"]["template"]["metadata"]["labels"]
        canary_sel = canary["spec"]["selector"]["matchLabels"]
        stable_sel = stable["spec"]["selector"]["matchLabels"]

        assert any(canary_pod_labels.get(k) != v for k, v in stable_sel.items()), "灰度 Pod 被稳定版选中"
        assert any(stable_pod_labels.get(k) != v for k, v in canary_sel.items()), "稳定版 Pod 被灰度选中"

    def test_owner_reference_for_cascade_delete(self) -> None:
        """没有 ownerReference 会导致删了 CR 资源还在，属于资源泄漏。"""
        spec = parse_spec({"model": "m"})
        dep = b.build_deployment("demo", "ns", spec, CR)
        owner = dep["metadata"]["ownerReferences"][0]
        assert owner["kind"] == KIND
        assert owner["uid"] == "uid-abc-123"
        assert owner["controller"] is True

    def test_no_owner_reference_without_uid(self) -> None:
        """单元测试/离线渲染场景没有 uid，此时不应写入无效的 ownerReference。"""
        spec = parse_spec({"model": "m"})
        dep = b.build_deployment("demo", "ns", spec, {"metadata": {"name": "demo"}})
        assert "ownerReferences" not in dep["metadata"]


class TestDeployment:
    def test_valid_for_kubernetes_model(self) -> None:
        """核心校验：生成的清单能被 K8s 官方模型反序列化。"""
        spec = parse_spec({"model": "m", "replicas": 3})
        dep = deserialize(b.build_deployment("demo", "ns", spec, CR), "V1Deployment")
        assert dep.metadata.name == "demo-gateway"
        assert dep.metadata.namespace == "ns"
        assert dep.spec.replicas == 3
        assert dep.spec.template.spec.containers[0].name == "gateway"

    def test_selector_matches_pod_template(self) -> None:
        """selector 与 Pod 模板标签不一致是 Deployment 创建失败的最常见原因。"""
        spec = parse_spec({"model": "m"})
        dep = b.build_deployment("demo", "ns", spec, CR)
        selector = dep["spec"]["selector"]["matchLabels"]
        pod_labels = dep["spec"]["template"]["metadata"]["labels"]
        assert all(pod_labels.get(k) == v for k, v in selector.items())

    def test_gpu_scheduling_when_gpu_requested(self) -> None:
        spec = parse_spec({"model": "m", "resources": {"gpu": 1}})
        dep = b.build_deployment("demo", "ns", spec, CR)
        pod = dep["spec"]["template"]["spec"]
        assert pod["nodeSelector"]["aetherserve.io/accelerator"] == "nvidia"
        assert pod["tolerations"][0]["key"] == "nvidia.com/gpu"
        limits = dep["spec"]["template"]["spec"]["containers"][0]["resources"]["limits"]
        assert limits["nvidia.com/gpu"] == "1"

    def test_no_gpu_scheduling_without_gpu(self) -> None:
        spec = parse_spec({"model": "m", "resources": {"gpu": 0}})
        dep = b.build_deployment("demo", "ns", spec, CR)
        assert "nodeSelector" not in dep["spec"]["template"]["spec"]

    def test_probes_configured(self) -> None:
        """readiness 走 /readyz（真实探测后端），liveness 走 /healthz（轻量）。"""
        spec = parse_spec({"model": "m"})
        container = b.build_deployment("demo", "ns", spec, CR)["spec"]["template"]["spec"]["containers"][0]
        assert container["readinessProbe"]["httpGet"]["path"] == "/readyz"
        assert container["livenessProbe"]["httpGet"]["path"] == "/healthz"
        assert container["startupProbe"]["failureThreshold"] == 30

    def test_spec_hash_annotation_triggers_rollout(self) -> None:
        s1 = parse_spec({"model": "m", "maxBatchSize": 8})
        s2 = parse_spec({"model": "m", "maxBatchSize": 16})
        h1 = b.build_deployment("demo", "ns", s1, CR)["spec"]["template"]["metadata"]["annotations"][
            "aetherserve.io/spec-hash"
        ]
        h2 = b.build_deployment("demo", "ns", s2, CR)["spec"]["template"]["metadata"]["annotations"][
            "aetherserve.io/spec-hash"
        ]
        assert h1 != h2

    def test_rolling_update_no_downtime(self) -> None:
        spec = parse_spec({"model": "m"})
        strategy = b.build_deployment("demo", "ns", spec, CR)["spec"]["strategy"]
        assert strategy["rollingUpdate"]["maxUnavailable"] == 0

    def test_graceful_shutdown_window(self) -> None:
        """优雅停机需要足够长的 grace period（引擎要排空队列）。"""
        spec = parse_spec({"model": "m"})
        dep = b.build_deployment("demo", "ns", spec, CR)
        assert dep["spec"]["template"]["spec"]["terminationGracePeriodSeconds"] >= 30

    def test_security_hardening(self) -> None:
        spec = parse_spec({"model": "m"})
        sc = b.build_deployment("demo", "ns", spec, CR)["spec"]["template"]["spec"]["containers"][0][
            "securityContext"
        ]
        assert sc["allowPrivilegeEscalation"] is False
        assert sc["runAsNonRoot"] is True
        assert sc["capabilities"]["drop"] == ["ALL"]

    def test_image_override_for_canary(self) -> None:
        spec = parse_spec({"model": "m"})
        dep = b.build_deployment("demo", "ns", spec, CR, canary=True, image="ghcr.io/x/gateway:v2")
        assert dep["spec"]["template"]["spec"]["containers"][0]["image"] == "ghcr.io/x/gateway:v2"

    def test_deterministic_output(self) -> None:
        """同样的 spec 必须生成完全相同的清单（幂等的基础）。"""
        spec = parse_spec({"model": "m", "replicas": 2})
        a = b.build_deployment("demo", "ns", spec, CR)
        c = b.build_deployment("demo", "ns", spec, CR)
        assert json.dumps(a, sort_keys=True) == json.dumps(c, sort_keys=True)


class TestService:
    def test_valid_for_kubernetes_model(self) -> None:
        spec = parse_spec({"model": "m"})
        svc = deserialize(b.build_service("demo", "ns", spec, CR), "V1Service")
        assert svc.metadata.name == "demo-stable"
        assert svc.spec.ports[0].port == 80
        # targetPort 在模型里是 target_port
        assert svc.spec.ports[0].target_port == "http"
        assert svc.spec.type == "ClusterIP"

    def test_service_selector_matches_deployment_pods(self) -> None:
        """Service 选不到 Pod 是"服务起来了但访问不通"的头号原因。"""
        spec = parse_spec({"model": "m"})
        dep = b.build_deployment("demo", "ns", spec, CR)
        svc = b.build_service("demo", "ns", spec, CR)
        pod_labels = dep["spec"]["template"]["metadata"]["labels"]
        assert all(pod_labels.get(k) == v for k, v in svc["spec"]["selector"].items())

    def test_canary_service_isolated(self) -> None:
        spec = parse_spec({"model": "m"})
        canary_dep = b.build_deployment("demo", "ns", spec, CR, canary=True)
        canary_svc = b.build_service("demo", "ns", spec, CR, canary=True)
        assert canary_svc["metadata"]["name"] == "demo-canary"
        pod_labels = canary_dep["spec"]["template"]["metadata"]["labels"]
        assert all(pod_labels.get(k) == v for k, v in canary_svc["spec"]["selector"].items())

    def test_canary_service_does_not_select_stable_pods(self) -> None:
        spec = parse_spec({"model": "m"})
        stable_pod_labels = b.build_deployment("demo", "ns", spec, CR)["spec"]["template"]["metadata"][
            "labels"
        ]
        canary_selector = b.build_service("demo", "ns", spec, CR, canary=True)["spec"]["selector"]
        assert any(stable_pod_labels.get(k) != v for k, v in canary_selector.items())


class TestHpa:
    def test_valid_for_kubernetes_model(self) -> None:
        spec = parse_spec({"model": "m", "autoscaling": {"minReplicas": 1, "maxReplicas": 8}})
        hpa = deserialize(b.build_hpa("demo", "ns", spec, CR), "V1HorizontalPodAutoscaler")
        assert hpa.spec.max_replicas == 8
        assert hpa.spec.scale_target_ref.kind == "Deployment"
        assert hpa.spec.scale_target_ref.name == "demo-gateway"

    def test_custom_metric_queue_depth(self) -> None:
        spec = parse_spec({"model": "m", "autoscaling": {"metrics": ["queue_depth"], "targetQueueDepth": 24}})
        hpa = b.build_hpa("demo", "ns", spec, CR)
        metric = hpa["spec"]["metrics"][0]
        assert metric["type"] == "Pods"
        assert metric["pods"]["metric"]["name"] == "aetherserve_queue_depth_avg"
        assert metric["pods"]["target"]["averageValue"] == "24"

    def test_multi_metric(self) -> None:
        spec = parse_spec({"model": "m", "autoscaling": {"metrics": ["queue_depth", "kv_cache", "cpu"]}})
        hpa = b.build_hpa("demo", "ns", spec, CR)
        assert len(hpa["spec"]["metrics"]) == 3

    def test_falls_back_to_cpu_when_no_metric(self) -> None:
        """HPA 必须至少有一个指标，否则 API Server 会拒绝。"""
        spec = parse_spec({"model": "m", "autoscaling": {"metrics": []}})
        hpa = b.build_hpa("demo", "ns", spec, CR)
        assert hpa["spec"]["metrics"][0]["type"] == "Resource"

    def test_scale_up_fast_scale_down_slow(self) -> None:
        """扩容要快、缩容要慢（避免抖动导致请求失败）。"""
        spec = parse_spec({"model": "m", "autoscaling": {"scaleDownStabilizationS": 600}})
        behavior = b.build_hpa("demo", "ns", spec, CR)["spec"]["behavior"]
        assert behavior["scaleUp"]["stabilizationWindowSeconds"] == 0
        assert behavior["scaleDown"]["stabilizationWindowSeconds"] == 600

    def test_max_replicas_never_below_min(self) -> None:
        spec = parse_spec({"model": "m", "autoscaling": {"minReplicas": 5, "maxReplicas": 1}})
        hpa = b.build_hpa("demo", "ns", spec, CR)
        assert hpa["spec"]["maxReplicas"] >= hpa["spec"]["minReplicas"]


class TestObservabilityManifests:
    def test_service_monitor(self) -> None:
        spec = parse_spec({"model": "m"})
        sm = b.build_service_monitor("demo", "ns", spec, CR)
        assert sm["kind"] == "ServiceMonitor"
        assert sm["spec"]["endpoints"][0]["path"] == "/metrics"
        assert sm["spec"]["namespaceSelector"]["matchNames"] == ["ns"]

    def test_service_monitor_selector_hits_service_labels(self) -> None:
        """ServiceMonitor 的 selector 必须能命中 Service 的标签，否则指标抓不到。"""
        spec = parse_spec({"model": "m"})
        svc_labels = b.build_service("demo", "ns", spec, CR)["metadata"]["labels"]
        match = b.build_service_monitor("demo", "ns", spec, CR)["spec"]["selector"]["matchLabels"]
        assert all(svc_labels.get(k) == v for k, v in match.items()), (
            f"ServiceMonitor 选不到 Service：match={match} svc_labels={svc_labels}"
        )

    def test_prometheus_rules_cover_key_failures(self) -> None:
        spec = parse_spec({"model": "m", "autoscaling": {"targetQueueDepth": 16}})
        rules = b.build_prometheus_rule("demo", "ns", spec, CR)
        alerts = {r["alert"] for r in rules["spec"]["groups"][0]["rules"]}
        assert {
            "AetherServeHighErrorRate",
            "AetherServeQueueBacklog",
            "AetherServeTTFTDegraded",
            "AetherServeCircuitBreakerOpen",
        } <= alerts

    def test_queue_backlog_threshold_derived_from_spec(self) -> None:
        """告警阈值应由 spec 推导，而非硬编码（否则改了目标值告警就失真）。"""
        spec = parse_spec({"model": "m", "autoscaling": {"targetQueueDepth": 25}})
        rules = b.build_prometheus_rule("demo", "ns", spec, CR)
        backlog = next(
            r for r in rules["spec"]["groups"][0]["rules"] if r["alert"] == "AetherServeQueueBacklog"
        )
        assert "> 100" in backlog["expr"]

    def test_pdb_only_when_replicas_gt_1(self) -> None:
        one = parse_spec({"model": "m", "replicas": 1})
        many = parse_spec({"model": "m", "replicas": 3})
        assert all(m["kind"] != "PodDisruptionBudget" for m in b.build_all("demo", "ns", one, CR))
        assert any(m["kind"] == "PodDisruptionBudget" for m in b.build_all("demo", "ns", many, CR))


class TestBuildAll:
    def test_default_bundle(self) -> None:
        spec = parse_spec({"model": "m", "replicas": 1, "serviceMonitor": False})
        kinds = [m["kind"] for m in b.build_all("demo", "ns", spec, CR)]
        assert kinds == ["Deployment", "Service", "HorizontalPodAutoscaler"]

    def test_autoscaling_disabled(self) -> None:
        spec = parse_spec({"model": "m", "replicas": 2, "autoscaling": {"enabled": False}})
        kinds = [m["kind"] for m in b.build_all("demo", "ns", spec, CR)]
        assert "HorizontalPodAutoscaler" not in kinds
        assert "PodDisruptionBudget" in kinds

    def test_canary_included(self) -> None:
        spec = parse_spec({"model": "m", "replicas": 2})
        kinds = [m["kind"] for m in b.build_all("demo", "ns", spec, CR, include_canary=True)]
        assert kinds.count("Deployment") == 2
        assert kinds.count("Service") == 2

    def test_all_unique_names(self) -> None:
        spec = parse_spec({"model": "m", "replicas": 2})
        ids = [b.manifest_identity(m) for m in b.build_all("demo", "ns", spec, CR, include_canary=True)]
        assert len(ids) == len(set(ids)), f"资源名重复：{ids}"

    def test_identities_are_stable(self) -> None:
        spec = parse_spec({"model": "m", "replicas": 2})
        a = [b.manifest_identity(m) for m in b.build_all("demo", "ns", spec, CR)]
        c = [b.manifest_identity(m) for m in b.build_all("demo", "ns", spec, CR)]
        assert a == c


class TestNameGeneration:
    def test_slugify_handles_model_paths(self) -> None:
        """模型名含 / 与 . ，必须被规范化为合法 DNS label。"""
        from aetherserve.operator.crd import slugify

        assert slugify("Qwen/Qwen2.5-1.5B-Instruct") == "qwen-qwen2-5-1-5b-instruct"
        assert slugify("meta-llama/Llama-3.1-8B") == "meta-llama-llama-3-1-8b"
        assert slugify("") == "x"
        assert slugify("---") == "x"

    def test_canonical_names(self) -> None:
        from aetherserve.operator.crd import canonical_name

        assert canonical_name("demo", "gateway") == "demo-gateway"
        assert canonical_name("demo") == "demo"


class TestManagedBySafety:
    def test_all_manifests_carry_managed_by_label(self) -> None:
        """孤儿资源清理依赖这个标签做安全检查，缺失会导致误删他人资源。"""
        spec = parse_spec({"model": "m", "replicas": 2})
        for m in b.build_all("demo", "ns", spec, CR, include_canary=True):
            labels = m["metadata"]["labels"]
            assert labels[b.LABEL_MANAGED_BY] == b.MANAGED_BY, f"{m['kind']} 缺少 managed-by 标签"
