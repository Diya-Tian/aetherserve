"""Kubernetes 资源清单构建器。

设计原则（Operator 的核心工程约束）
----------------------------------
1. **纯函数**：``build_*`` 只依赖 ``InferenceServiceSpec``，不读集群、不写集群。
   这让"生成什么"与"如何调和"彻底分离，前者可以用单元测试覆盖到 100%；
2. **幂等**：同样的 spec 必须生成**逐字节相同**的清单。
   为此所有 dict 的键顺序固定、不使用随机值、时间戳不写入模板（只写入 annotation 的 spec hash）；
3. **不可变字段保护**：Deployment 的 ``selector`` 一旦创建不可修改，
   因此 selector 只由 ``CR 名称`` 决定，**绝不含 spec hash**，否则第二次调和就会失败；
4. **变更可触发**：把 ``spec hash`` 写进 Pod 模板 annotation，
   使镜像/环境变量/资源变化能够触发滚动更新，而不相关的调和不会重启 Pod。

生成清单的验证方式：测试中会用 ``kubernetes`` 官方客户端的反序列化器
把每个清单解析成 ``V1Deployment`` / ``V1Service`` / ``V1HorizontalPodAutoscaler``
等类型化模型 —— 能通过说明清单结构合法（比"能 dump 成 YAML"强得多）。
"""

from __future__ import annotations

from typing import Any

from .crd import (
    API_GROUP,
    API_VERSION,
    KIND,
    InferenceServiceSpec,
    canonical_name,
)

# 标签键：保持稳定，监控与选择器都依赖它
LABEL_APP = "app.kubernetes.io/name"
LABEL_INSTANCE = "app.kubernetes.io/instance"
LABEL_MANAGED_BY = "app.kubernetes.io/managed-by"
LABEL_COMPONENT = "app.kubernetes.io/component"
LABEL_CR = "aetherserve.io/cr"
# 版本角色：stable | canary
# 用专属键而不是复用 app.kubernetes.io/* —— 语义独立，避免与生态工具的标签约定互相干扰
LABEL_ROLE = "aetherserve.io/role"

MANAGED_BY = "aetherserve-operator"
GATEWAY_COMPONENT = "gateway"
ROLE_STABLE = "stable"
ROLE_CANARY = "canary"


# --------------------------------------------------------------------- 标签
def role_for(canary: bool) -> str:
    return ROLE_CANARY if canary else ROLE_STABLE


def labels_for(
    name: str,
    role: str = ROLE_STABLE,
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    """生成 Pod / 资源统一使用的标签集合。

    标签语义分工（这套约定是"稳定版与灰度版互不串流量"的基础）
    ---------------------------------------------------------
    - ``app.kubernetes.io/name``      = ``aetherserve``（产品名）
    - ``app.kubernetes.io/instance``  = **CR 名称**（业务实例，不含版本信息）
    - ``app.kubernetes.io/component`` = ``gateway``（组件，稳定版与灰度版相同）
    - ``aetherserve.io/role``         = ``stable`` / ``canary``（**唯一的版本判别维度**）
    - ``aetherserve.io/cr``           = CR 名称（供 ServiceMonitor 做交集匹配）
    - ``app.kubernetes.io/managed-by``= ``aetherserve-operator``（孤儿资源清理的安全依据）

    为什么版本必须用独立标签表达？
    如果把版本塞进 ``instance``（例如 ``demo-stable``），会同时踩两个坑：
    1. Service 与 Deployment 的标签容易写得不一致 → **Service 选不到 Pod，服务访问不通**；
    2. 稳定版 selector 若只匹配 ``instance``，就会把灰度 Pod 一起选中 → **灰度流量失控**。
    """
    out = {
        LABEL_APP: "aetherserve",
        LABEL_INSTANCE: name,
        LABEL_COMPONENT: GATEWAY_COMPONENT,
        LABEL_ROLE: role,
        LABEL_CR: name,
        LABEL_MANAGED_BY: MANAGED_BY,
    }
    if extra:
        out.update(extra)
    return out


def selector_labels(name: str, role: str = ROLE_STABLE) -> dict[str, str]:
    """Deployment 与 Service **共用**的选择器标签。

    **不可变约束**：本函数只依赖「CR 名称 + 角色」，绝不包含 spec hash。
    K8s 禁止修改 Deployment 的 ``spec.selector``，一旦把可变内容放进来，
    第二次调和就会被 API Server 拒绝。版本角色是固定字面量，因此同样不可变。
    """
    return {LABEL_APP: "aetherserve", LABEL_INSTANCE: name, LABEL_ROLE: role}


def service_selector(name: str, canary: bool) -> dict[str, str]:
    """Service 的 selector —— 与对应 Deployment 的 selector **同一个函数产出**。

    这样从结构上消除"两者写得不一致导致 Service 选不到 Pod"的可能性。
    """
    return selector_labels(name, role_for(canary))


def owner_reference(cr: dict[str, Any]) -> dict[str, Any]:
    """构造 ownerReference，实现级联回收：删除 CR 时子资源自动清理。

    缺少它会导致"删了 CR 但 Deployment 还在跑"，是新手 Operator 最常见的资源泄漏。
    """
    metadata = cr.get("metadata", {})
    return {
        "apiVersion": f"{API_GROUP}/{API_VERSION}",
        "kind": KIND,
        "name": metadata.get("name", ""),
        "uid": metadata.get("uid", ""),
        "controller": True,
        "blockOwnerDeletion": True,
    }


def _base_metadata(
    name: str,
    namespace: str,
    cr: dict[str, Any],
    role: str = ROLE_STABLE,
    extra_labels: dict[str, str] | None = None,
) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "name": name,
        "namespace": namespace,
        "labels": labels_for(cr.get("metadata", {}).get("name", name), role, extra_labels),
    }
    owner = owner_reference(cr)
    if owner["uid"]:  # 单元测试中可能没有 uid，此时不写 ownerReference
        meta["ownerReferences"] = [owner]
    return meta


def _probe(path: str, port: int, initial_delay: int, period: int = 10) -> dict[str, Any]:
    return {
        "httpGet": {"path": path, "port": port},
        "initialDelaySeconds": initial_delay,
        "periodSeconds": period,
        "timeoutSeconds": 3,
        "failureThreshold": 3,
    }


# --------------------------------------------------------------------- Deployment
def build_deployment(
    name: str,
    namespace: str,
    spec: InferenceServiceSpec,
    cr: dict[str, Any],
    *,
    canary: bool = False,
    replicas: int | None = None,
    image: str | None = None,
) -> dict[str, Any]:
    """生成推理网关 Deployment。

    Args:
        canary: 是否生成灰度版本（名称、标签、副本策略都不同）。
        replicas: 覆盖副本数（灰度版本常与稳定版不同）。
        image: 覆盖镜像（灰度通常发布新镜像）。
    """
    role = role_for(canary)
    res_name = canonical_name(name, "canary" if canary else "gateway")
    image_ref = image or spec.image
    desired_replicas = replicas if replicas is not None else spec.replicas

    # 环境变量列表：值可能是 str，也可能是 valueFrom 引用，故元素类型为 dict[str, Any]
    env_list: list[dict[str, Any]] = [{"name": k, "value": v} for k, v in sorted(spec.gateway_env().items())]
    env_list.append({"name": "POD_NAME", "valueFrom": {"fieldRef": {"fieldPath": "metadata.name"}}})

    container: dict[str, Any] = {
        "name": "gateway",
        "image": image_ref,
        "imagePullPolicy": "IfNotPresent",
        "ports": [
            {"name": "http", "containerPort": spec.port, "protocol": "TCP"},
        ],
        "env": env_list,
        "resources": {"requests": spec.resources.requests(), "limits": spec.resources.limits()},
        # readiness 走 /readyz（会真实探测后端），保证"没准备好就不接流量"
        "readinessProbe": _probe("/readyz", spec.port, initial_delay=5),
        # liveness 走 /healthz（只查进程与引擎状态），避免后端抖动导致 Pod 被重启
        "livenessProbe": _probe("/healthz", spec.port, initial_delay=15, period=20),
        "startupProbe": {
            "httpGet": {"path": "/healthz", "port": spec.port},
            "failureThreshold": 30,
            "periodSeconds": 5,
        },
        "securityContext": {
            "allowPrivilegeEscalation": False,
            "runAsNonRoot": True,
            "runAsUser": 10001,
            "capabilities": {"drop": ["ALL"]},
        },
    }

    pod_labels = labels_for(name, role, {"aetherserve.io/spec-hash": spec.hash()})
    pod_spec: dict[str, Any] = {
        "containers": [container],
        "terminationGracePeriodSeconds": 60,  # 给优雅停机留足时间（引擎会排空队列）
        "affinity": {
            "podAntiAffinity": {
                "preferredDuringSchedulingIgnoredDuringExecution": [
                    {
                        "weight": 100,
                        "podAffinityTerm": {
                            "topologyKey": "kubernetes.io/hostname",
                            # 仅针对同角色的副本做反亲和（稳定版与灰度版本就应该能同节点共存）
                            "labelSelector": {"matchLabels": selector_labels(name, role)},
                        },
                    }
                ]
            }
        },
    }

    if spec.resources.gpu > 0:
        # GPU 节点隔离：只在带 GPU 标签的节点上调度，避免占用通用节点
        node_selector = {"aetherserve.io/accelerator": "nvidia", **(spec.node_selector or {})}
        pod_spec["nodeSelector"] = node_selector
        pod_spec["tolerations"] = [{"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}]
    elif spec.node_selector:
        pod_spec["nodeSelector"] = dict(spec.node_selector)

    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": _base_metadata(
            res_name,
            namespace,
            cr,
            role,
            extra_labels={"aetherserve.io/spec-hash": spec.hash()},
        ),
        "spec": {
            "replicas": desired_replicas,
            # 选择器必须"创建后永不变"：只由 CR 名 + 角色决定，不含 spec hash。
            # 若把 hash 放进 selector，第二次调和就会因"selector 不可变"而被 API 拒绝。
            "selector": {"matchLabels": selector_labels(name, role)},
            "strategy": {
                "type": "RollingUpdate",
                "rollingUpdate": {"maxUnavailable": 0, "maxSurge": 1},
            },
            "revisionHistoryLimit": 5,
            "template": {
                "metadata": {
                    "labels": pod_labels,
                    "annotations": {
                        # 只有这个哈希变化才会滚动更新 Pod
                        "aetherserve.io/spec-hash": spec.hash(),
                        "prometheus.io/scrape": "true",
                        "prometheus.io/port": str(spec.port),
                        "prometheus.io/path": "/metrics",
                    },
                },
                "spec": pod_spec,
            },
        },
    }


# --------------------------------------------------------------------- Service
def build_service(
    name: str,
    namespace: str,
    spec: InferenceServiceSpec,
    cr: dict[str, Any],
    *,
    canary: bool = False,
) -> dict[str, Any]:
    """生成 Service（ClusterIP）。"""
    role = role_for(canary)
    res_name = canonical_name(name, role)

    return {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": _base_metadata(
            res_name,
            namespace,
            cr,
            role,
            # 额外打上 CR 名标签，供 ServiceMonitor 做交集匹配
            extra_labels={"aetherserve.io/cr": name},
        ),
        "spec": {
            "type": "ClusterIP",
            # 与对应 Deployment 的 selector 出自同一函数 → 结构上不可能不一致。
            # 这是"服务起得来但访问不通"这类事故的根治手段。
            "selector": service_selector(name, canary),
            "ports": [
                {
                    "name": "http",
                    "port": 80,
                    "targetPort": "http",
                    "protocol": "TCP",
                }
            ],
        },
    }


# --------------------------------------------------------------------- HPA
def build_hpa(
    name: str,
    namespace: str,
    spec: InferenceServiceSpec,
    cr: dict[str, Any],
) -> dict[str, Any]:
    """生成 autoscaling/v2 HPA（支持自定义指标）。

    指标选择说明
    ------------
    - ``queue_depth``：用 **Pods** 类型的 AverageValue 指标
      （每个 Pod 的平均队列深度），它反映"单副本压力"，扩缩容最直接；
    - ``kv_cache``：用 Pods 类型指标表达 KV Cache 水位。

    HPA 通过 Prometheus Adapter 把这些自定义指标暴露给 ``custom.metrics.k8s.io``。
    """
    auto = spec.autoscaling
    metrics: list[dict[str, Any]] = []

    for metric in auto.metrics:
        if metric == "queue_depth":
            metrics.append(
                {
                    "type": "Pods",
                    "pods": {
                        "metric": {"name": "aetherserve_queue_depth_avg"},
                        "target": {"type": "AverageValue", "averageValue": str(auto.target_queue_depth)},
                    },
                }
            )
        elif metric == "kv_cache":
            # KV Cache 水位是 0~1 的比例，改用 Utilization 语义更贴切
            metrics.append(
                {
                    "type": "Pods",
                    "pods": {
                        "metric": {"name": "aetherserve_kv_cache_utilization"},
                        "target": {
                            "type": "AverageValue",
                            "averageValue": f"{int(auto.target_kv_cache_utilization * 100)}m",
                        },
                    },
                }
            )
        elif metric == "cpu":
            metrics.append(
                {
                    "type": "Resource",
                    "resource": {"name": "cpu", "target": {"type": "Utilization", "averageUtilization": 70}},
                }
            )

    if not metrics:  # 兜底：HPA 必须至少有一个指标，否则 API 会拒绝
        metrics.append(
            {
                "type": "Resource",
                "resource": {"name": "cpu", "target": {"type": "Utilization", "averageUtilization": 70}},
            }
        )

    return {
        "apiVersion": "autoscaling/v2",
        "kind": "HorizontalPodAutoscaler",
        "metadata": _base_metadata(
            canonical_name(name, "hpa"), namespace, cr, ROLE_STABLE, {"aetherserve.io/kind": "autoscaler"}
        ),
        "spec": {
            "scaleTargetRef": {
                "apiVersion": "apps/v1",
                "kind": "Deployment",
                "name": canonical_name(name, "gateway"),
            },
            "minReplicas": max(auto.min_replicas, 1) if spec.replicas == 0 else auto.min_replicas,
            "maxReplicas": max(auto.max_replicas, auto.min_replicas, 1),
            "metrics": metrics,
            "behavior": {
                # 扩容要快（应对流量尖峰）：无稳定窗口，最多翻倍
                "scaleUp": {
                    "stabilizationWindowSeconds": 0,
                    "policies": [{"type": "Percent", "value": 100, "periodSeconds": 30}],
                },
                # 缩容要慢（避免抖动导致请求失败）：默认 5 分钟稳定窗口
                "scaleDown": {
                    "stabilizationWindowSeconds": auto.scale_down_stabilization_s,
                    "policies": [{"type": "Percent", "value": 50, "periodSeconds": 60}],
                },
            },
        },
    }


# --------------------------------------------------------------------- 观测
def build_service_monitor(
    name: str,
    namespace: str,
    spec: InferenceServiceSpec,
    cr: dict[str, Any],
) -> dict[str, Any]:
    """生成 Prometheus Operator 的 ServiceMonitor，让指标被自动抓取。"""
    # ServiceMonitor 的 selector 必须精确命中 Service 的标签。
    # 注意不能用 component（各资源取值不同），改用 aetherserve.io/cr 做交集匹配：
    # 该标签在所有子资源上都等于 CR 名，因此能唯一锁定"本 CR 的网关 Service"。
    return {
        "apiVersion": "monitoring.coreos.com/v1",
        "kind": "ServiceMonitor",
        "metadata": _base_metadata(
            canonical_name(name, "monitor"), namespace, cr, ROLE_STABLE, {"aetherserve.io/kind": "monitoring"}
        ),
        "spec": {
            "selector": {
                "matchLabels": {
                    # 精确命中本 CR 的稳定版 Service 标签（与 build_service 的标签一致）
                    LABEL_APP: "aetherserve",
                    LABEL_CR: name,
                }
            },
            "namespaceSelector": {"matchNames": [namespace]},
            "endpoints": [
                {
                    "port": "http",
                    "path": "/metrics",
                    "interval": "15s",
                    "scrapeTimeout": "10s",
                }
            ],
        },
    }


def build_prometheus_rule(
    name: str,
    namespace: str,
    spec: InferenceServiceSpec,
    cr: dict[str, Any],
) -> dict[str, Any]:
    """生成 Prometheus 告警规则。

    告警口径来自真实排障经验，而不是"随便加个阈值"：
    - ``HighErrorRate``：错误率持续 5 分钟 > 5%，通常是后端 OOM 或模型加载失败；
    - ``QueueBacklog``：队列深度持续 5 分钟 > 目标值 4 倍，说明扩容跟不上或后端卡死；
    - ``TTFTDegraded``：首 token 延迟 P99 超过 5s，用户体验已不可接受；
    - ``CircuitBreakerOpen``：熔断器打开，后端已不可用（快速失败正在生效）。
    """
    res_name = canonical_name(name, "gateway")
    job_label = f'job="{res_name}"'

    return {
        "apiVersion": "monitoring.coreos.com/v1",
        "kind": "PrometheusRule",
        "metadata": _base_metadata(
            canonical_name(name, "rules"), namespace, cr, ROLE_STABLE, {"aetherserve.io/kind": "monitoring"}
        ),
        "spec": {
            "groups": [
                {
                    "name": f"{res_name}.rules",
                    "interval": "30s",
                    "rules": [
                        {
                            "alert": "AetherServeHighErrorRate",
                            "expr": (
                                f'sum(rate(aetherserve_requests_total{{status="error",{job_label}}}[5m]))'
                                f" / sum(rate(aetherserve_requests_total{{{job_label}}}[5m])) > 0.05"
                            ),
                            "for": "5m",
                            "labels": {"severity": "critical", "service": res_name},
                            "annotations": {
                                "summary": "推理服务错误率超过 5%",
                                "description": "{{ $labels.instance }} 错误率 {{ $value | humanizePercentage }}",
                            },
                        },
                        {
                            "alert": "AetherServeQueueBacklog",
                            "expr": (
                                f"avg(aetherserve_queue_depth{{{job_label}}})"
                                f" > {spec.autoscaling.target_queue_depth * 4}"
                            ),
                            "for": "5m",
                            "labels": {"severity": "warning", "service": res_name},
                            "annotations": {
                                "summary": "推理队列积压，扩容或后端可能异常",
                                "description": "队列深度 {{ $value }}，目标值 "
                                f"{spec.autoscaling.target_queue_depth}",
                            },
                        },
                        {
                            "alert": "AetherServeTTFTDegraded",
                            "expr": (
                                f"histogram_quantile(0.99, sum(rate("
                                f"aetherserve_ttft_seconds_bucket{{{job_label}}}[5m])) by (le)) > 5"
                            ),
                            "for": "5m",
                            "labels": {"severity": "warning", "service": res_name},
                            "annotations": {
                                "summary": "首 token 延迟 P99 超过 5 秒",
                                "description": "TTFT P99 = {{ $value }}s",
                            },
                        },
                        {
                            "alert": "AetherServeCircuitBreakerOpen",
                            "expr": f"max(aetherserve_circuit_breaker_state{{{job_label}}}) >= 2",
                            "for": "1m",
                            "labels": {"severity": "critical", "service": res_name},
                            "annotations": {
                                "summary": "推理后端熔断器已打开",
                                "description": "后端 {{ $labels.backend }} 连续失败已触发熔断",
                            },
                        },
                    ],
                }
            ]
        },
    }


def build_pdb(
    name: str,
    namespace: str,
    spec: InferenceServiceSpec,
    cr: dict[str, Any],
) -> dict[str, Any]:
    """生成 PodDisruptionBudget：节点维护时不把副本一起驱逐光。"""
    return {
        "apiVersion": "policy/v1",
        "kind": "PodDisruptionBudget",
        "metadata": _base_metadata(
            canonical_name(name, "pdb"), namespace, cr, ROLE_STABLE, {"aetherserve.io/kind": "availability"}
        ),
        "spec": {
            "minAvailable": 1,
            "selector": {"matchLabels": selector_labels(name, ROLE_STABLE)},
        },
    }


# --------------------------------------------------------------------- 聚合
def build_all(
    name: str,
    namespace: str,
    spec: InferenceServiceSpec,
    cr: dict[str, Any],
    *,
    include_canary: bool = False,
) -> list[dict[str, Any]]:
    """构建一个 InferenceService 需要的全部清单（调和的"期望状态"）。

    返回顺序固定：Deployment → Service → (HPA) → (Monitor) → (Rules) → (PDB)。
    顺序稳定让"实际状态与期望状态比对"可以直接按索引进行，简化调和逻辑。
    """
    manifests: list[dict[str, Any]] = [
        build_deployment(name, namespace, spec, cr),
        build_service(name, namespace, spec, cr),
    ]

    if include_canary:
        manifests.append(build_deployment(name, namespace, spec, cr, canary=True))
        manifests.append(build_service(name, namespace, spec, cr, canary=True))

    if spec.autoscaling.enabled:
        manifests.append(build_hpa(name, namespace, spec, cr))

    if spec.service_monitor:
        manifests.append(build_service_monitor(name, namespace, spec, cr))
        manifests.append(build_prometheus_rule(name, namespace, spec, cr))

    if spec.replicas > 1:
        manifests.append(build_pdb(name, namespace, spec, cr))

    return manifests


def manifest_identity(manifest: dict[str, Any]) -> tuple[str, str, str]:
    """返回资源的唯一标识 ``(apiVersion, kind, name)``，用于比对期望/实际状态。"""
    return (
        str(manifest.get("apiVersion", "")),
        str(manifest.get("kind", "")),
        str(manifest.get("metadata", {}).get("name", "")),
    )


def existing_spec_hash(resource: dict[str, Any]) -> str | None:
    """从已存在的资源中读出 spec hash（判断是否需要更新）。"""
    # 优先看 Pod 模板 annotation（Deployment），否则看资源自身 labels
    template = resource.get("spec", {}).get("template", {})
    annotations = template.get("metadata", {}).get("annotations", {}) or {}
    if "aetherserve.io/spec-hash" in annotations:
        return str(annotations["aetherserve.io/spec-hash"])
    labels = resource.get("metadata", {}).get("labels", {}) or {}
    if "aetherserve.io/spec-hash" in labels:
        return str(labels["aetherserve.io/spec-hash"])
    return None


def owner_uids(resource: dict[str, Any]) -> set[str]:
    """取出资源的所有 ownerReference uid。

    调和循环用它判定"这个资源是否归本 CR 所有"：
    比按名字子串匹配可靠得多（名字可能重合），也不会误删其他 CR 的资源。
    """
    refs = resource.get("metadata", {}).get("ownerReferences", []) or []
    return {str(r.get("uid")) for r in refs if r.get("uid")}


def owned_by(resource: dict[str, Any], uid: str | None) -> bool:
    """判断资源是否由指定 CR 拥有。

    宽容策略：无法取得 uid 时（例如离线单测、本地渲染）退化为"按 managed-by 标签归属"，
    这样在缺少集群元数据的情况下调和逻辑依然可用。
    """
    if not uid:
        labels = resource.get("metadata", {}).get("labels", {}) or {}
        return labels.get(LABEL_MANAGED_BY) == MANAGED_BY
    return uid in owner_uids(resource)
