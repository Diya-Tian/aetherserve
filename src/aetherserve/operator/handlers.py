"""kopf 调和处理器 —— 把 ``reconciler`` 的计划写回集群。

职责边界（刻意的薄封装）
----------------------
``reconciler.plan_reconcile`` 已经算出"该创建/更新/删除什么"，
本模块只做三件事：**读集群 → 调计划 → 写集群**。
因此这里几乎没有业务分支，出错面很小。

kopf 的关键概念
--------------
- ``@kopf.on.startup`` / ``@kopf.on.cleanup``：进程生命周期钩子；
- ``@kopf.on.create`` / ``@kopf.on.update`` / ``@kopf.on.resume``：
  分别对应"对象新建 / spec 变更 / Operator 重启后重新接管已有对象"。
  **``resume`` 极易被漏掉** —— 没有它，Operator 重启后已存在的 CR 永远不会被重新调和；
- ``@kopf.timer``：周期性调和（用于灰度分析这类"时间驱动"的逻辑）；
- ``@kopf.on.delete``：级联清理（本项目用 ownerReference 自动回收，无需手写）。

写操作的 API 选择
----------------
用 ``patch``（strategic merge）而非 ``update``：
- update 会覆盖整个对象，多个控制器同时写会互相踩踏；
- patch 只提交我们关心的字段，且天然幂等；
- 首次创建用 ``create``，之后统一走 ``patch``。
"""

from __future__ import annotations

import logging
from typing import Any

from ..observability import metrics
from .builders import LABEL_MANAGED_BY, MANAGED_BY, owned_by
from .canary import CanaryMetrics
from .crd import API_GROUP, API_VERSION, KIND, PLURAL
from .reconciler import ReconcilePlan, plan_reconcile

log = logging.getLogger("aetherserve.operator")


# --------------------------------------------------------------------- 客户端
def _load_clients() -> tuple[Any, Any, Any]:
    """惰性加载 Kubernetes 客户端，避免导入时就要求集群凭据存在。"""
    from kubernetes import client, config

    try:
        config.load_incluster_config()  # 运行在集群内（ServiceAccount）
    except Exception:
        config.load_kube_config()  # 本地开发（~/.kube/config）
    return client.AppsV1Api(), client.CoreV1Api(), client.AutoscalingV2Api()


def _deserialize(manifest: dict[str, Any], model: str) -> Any:
    """把清单 dict 反序列化成 Kubernetes 类型化模型。

    说明：客户端库只提供两个入口 —— 公开的 ``deserialize(response, model)``
    需要 ``urllib3.HTTPResponse``（面向真实 HTTP 响应），
    以及面向 dict 的 ``_ApiClient__deserialize(data, model)``（名称改写后的私有方法）。
    我们手上是刚生成的清单 dict，因此只能用后者；这里集中封装一次并加 noqa，
    避免私有调用散落各处（升级客户端库时只需改这一处）。
    """
    from kubernetes import client

    return client.ApiClient()._ApiClient__deserialize(manifest, model)


def _api_for(kind: str) -> str:
    """按 Kind 选择写操作要走哪个 API 分组。"""
    mapping = {
        "Deployment": "apps",
        "Service": "core",
        "HorizontalPodAutoscaler": "autoscaling",
        "PodDisruptionBudget": "policy",
        "ServiceMonitor": "custom",
        "PrometheusRule": "custom",
    }
    return mapping.get(kind, "custom")


def _create_or_patch(manifest: dict[str, Any]) -> str:
    """创建或打补丁（幂等的关键：已存在就走 patch，不报错）。

    Returns:
        ``"created"`` / ``"patched"`` / ``"skipped"``
    """
    from kubernetes import client
    from kubernetes.client.exceptions import ApiException

    kind = manifest.get("kind", "")
    name = manifest.get("metadata", {}).get("name", "")
    namespace = manifest.get("metadata", {}).get("namespace", "default")
    group = _api_for(kind)

    try:
        # 每个分支使用独立的局部变量名：复用一个变量会让类型检查器推断出首个分支的类型，
        # 从而把后续分支的方法调用全部判为错误。
        if group == "apps":
            apps_api = client.AppsV1Api()
            body = _deserialize(manifest, "V1Deployment")
            try:
                apps_api.create_namespaced_deployment(namespace, body)
                return "created"
            except ApiException as exc:
                if exc.status != 409:
                    raise
                apps_api.patch_namespaced_deployment(name, namespace, manifest)
                return "patched"

        if group == "core":
            core_api = client.CoreV1Api()
            body = _deserialize(manifest, "V1Service")
            try:
                core_api.create_namespaced_service(namespace, body)
                return "created"
            except ApiException as exc:
                if exc.status != 409:
                    raise
                # Service 的 clusterIP 等字段不可变，patch 时需剔除 status
                safe = {k: v for k, v in manifest.items() if k != "status"}
                core_api.patch_namespaced_service(name, namespace, safe)
                return "patched"

        if group == "autoscaling":
            autoscaling_api = client.AutoscalingV2Api()
            body = _deserialize(manifest, "V1HorizontalPodAutoscaler")
            try:
                autoscaling_api.create_namespaced_horizontal_pod_autoscaler(namespace, body)
                return "created"
            except ApiException as exc:
                if exc.status != 409:
                    raise
                autoscaling_api.patch_namespaced_horizontal_pod_autoscaler(name, namespace, manifest)
                return "patched"

        if group == "policy":
            policy_api = client.PolicyV1Api()
            try:
                policy_api.create_namespaced_pod_disruption_budget(
                    namespace, _deserialize(manifest, "V1PodDisruptionBudget")
                )
                return "created"
            except ApiException as exc:
                if exc.status != 409:
                    raise
                policy_api.patch_namespaced_pod_disruption_budget(name, namespace, manifest)
                return "patched"

        # CRD 类型（ServiceMonitor / PrometheusRule）：用动态客户端
        from kubernetes import dynamic

        dyn = dynamic.DynamicClient(client.ApiClient())
        api_version = manifest.get("apiVersion", "")
        resource = dyn.resources.get(api_version=api_version, kind=kind)
        try:
            resource.create(body=manifest, namespace=namespace)
            return "created"
        except ApiException as exc:
            if exc.status != 409:
                raise
            resource.patch(
                body=manifest, name=name, namespace=namespace, content_type="application/merge-patch+json"
            )
            return "patched"

    except ApiException as exc:
        log.error("apply_failed kind=%s name=%s status=%s reason=%s", kind, name, exc.status, exc.reason)
        metrics.BACKEND_ERRORS.labels(backend="operator", kind=f"apply_{exc.status}").inc()
        return "skipped"


def _delete_resource(resource: dict[str, Any]) -> bool:
    """删除孤儿资源（灰度回滚后清理 canary 版本）。"""
    from kubernetes import client
    from kubernetes.client.exceptions import ApiException

    kind = resource.get("kind", "")
    name = resource.get("metadata", {}).get("name", "")
    namespace = resource.get("metadata", {}).get("namespace", "default")
    try:
        if kind == "Deployment":
            client.AppsV1Api().delete_namespaced_deployment(name, namespace)
        elif kind == "Service":
            client.CoreV1Api().delete_namespaced_service(name, namespace)
        elif kind == "HorizontalPodAutoscaler":
            client.AutoscalingV2Api().delete_namespaced_horizontal_pod_autoscaler(name, namespace)
        elif kind == "PodDisruptionBudget":
            client.PolicyV1Api().delete_namespaced_pod_disruption_budget(name, namespace)
        else:
            return False
        log.info("orphan_deleted kind=%s name=%s", kind, name)
        return True
    except ApiException as exc:
        if exc.status not in (404, 409):
            log.error("delete_failed kind=%s name=%s status=%s", kind, name, exc.status)
        return False


# --------------------------------------------------------------------- 观测
def _list_owned(namespace: str, name: str, uid: str | None = None) -> list[dict[str, Any]]:
    """列出由该 CR 拥有的子资源（供求差使用）。

    归属判断优先用 **ownerReference uid**，而不是按资源名做子串匹配：
    后者在多个 CR 名字互为前缀时会张冠李戴，进而误删别人的资源。
    """
    from kubernetes import client

    found: list[dict[str, Any]] = []
    api_client = client.ApiClient()
    selector = f"{LABEL_MANAGED_BY}={MANAGED_BY}"

    def _consider(obj: Any) -> None:
        raw = api_client.sanitize_for_serialization(obj)
        if owned_by(raw, uid):
            found.append(raw)

    try:
        for dep in client.AppsV1Api().list_namespaced_deployment(namespace, label_selector=selector).items:
            _consider(dep)
        for svc in client.CoreV1Api().list_namespaced_service(namespace, label_selector=selector).items:
            _consider(svc)
        for hpa in (
            client.AutoscalingV2Api()
            .list_namespaced_horizontal_pod_autoscaler(namespace, label_selector=selector)
            .items
        ):
            _consider(hpa)
        for pdb in (
            client.PolicyV1Api()
            .list_namespaced_pod_disruption_budget(namespace, label_selector=selector)
            .items
        ):
            _consider(pdb)
    except Exception as exc:  # pragma: no cover - 需要真实集群
        log.warning("list_owned_failed namespace=%s name=%s err=%s", namespace, name, exc)
    return found


def _fetch_canary_metrics(namespace: str, name: str) -> CanaryMetrics:
    """从 Prometheus 查询灰度版本的错误率与请求量。

    在真实部署里这是对 Prometheus HTTP API 的一次查询；这里做了异常兜底：
    监控不可用时返回"样本不足"，从而让灰度**停在原地**而不是盲目推进 ——
    这是关键的安全默认值（fail-safe 而非 fail-open）。
    """
    import os

    prom = os.environ.get("AETHERSERVE_PROMETHEUS_URL")
    if not prom:
        return CanaryMetrics()

    try:
        import httpx

        # 与 ServiceMonitor 的 selector 对齐：用 aetherserve.io/cr 标签精确定位本 CR 的稳定版
        label = f'aetherserve_io_cr="{name}"'
        total_q = f"sum(increase(aetherserve_requests_total{{{label}}}[5m]))"
        error_q = f'sum(increase(aetherserve_requests_total{{{label},status="error"}}[5m]))'
        with httpx.Client(timeout=5.0) as c:
            total = c.get(f"{prom}/api/v1/query", params={"query": total_q}).json()
            errors = c.get(f"{prom}/api/v1/query", params={"query": error_q}).json()
        t = float(total["data"]["result"][0]["value"][1]) if total["data"]["result"] else 0.0
        e = float(errors["data"]["result"][0]["value"][1]) if errors["data"]["result"] else 0.0
        return CanaryMetrics(request_count=int(t), error_count=int(e))
    except Exception as exc:
        log.warning("prometheus_query_failed name=%s err=%s", name, exc)
        return CanaryMetrics()  # fail-safe：样本不足 → 灰度不推进


def _apply_plan(plan: ReconcilePlan) -> dict[str, Any]:
    """执行计划并把结果汇总（供 kopf 写入 status）。"""
    results: dict[str, str] = {}
    for action in plan.actions:
        _, kind, res_name = action.identity
        key = f"{kind}/{res_name}"
        if action.op in ("create", "update"):
            results[key] = _create_or_patch(action.manifest)
        elif action.op == "delete":
            results[key] = "deleted" if _delete_resource(action.manifest) else "delete_skipped"
        else:
            results[key] = "noop"
    return results


# --------------------------------------------------------------------- kopf 处理器
def register_handlers() -> None:
    """把所有处理器注册到 kopf（由 ``main.py`` 调用）。

    拆成函数而不是模块级装饰器，是为了让测试在**不启动 kopf**的情况下
    直接导入本模块（模块级 ``@kopf.on.*`` 会要求 kopf 运行上下文）。
    """
    import kopf

    @kopf.on.startup()
    async def on_startup(**_kwargs: Any) -> None:
        log.info("aetherserve-operator 启动，开始监听 %s/%s", API_GROUP, KIND)
        metrics.CIRCUIT_STATE.labels(backend="operator").set(0.0)

    @kopf.on.cleanup()
    async def on_cleanup(**_kwargs: Any) -> None:
        log.info("aetherserve-operator 退出")

    @kopf.on.create(API_GROUP, API_VERSION, PLURAL)
    @kopf.on.update(API_GROUP, API_VERSION, PLURAL)
    @kopf.on.resume(API_GROUP, API_VERSION, PLURAL)
    async def reconcile(
        spec: dict[str, Any], meta: dict[str, Any], body: dict[str, Any], **_kwargs: Any
    ) -> dict[str, Any]:
        """核心调和处理器：create / update / resume 共用同一段逻辑。

        共用是正确的：调和本来就该是"无论怎么被触发，都让实际状态收敛到期望状态"。
        """
        import time

        name = meta.get("name", "")
        namespace = meta.get("namespace", "default")
        started = time.monotonic()

        with metrics.RECONCILE_DURATION.labels(resource="inferenceservice").time():
            try:
                existing = _list_owned(namespace, name)
                cr = {"metadata": meta, "spec": spec, "status": body.get("status", {})}
                plan = plan_reconcile(cr, existing, canary_metrics=_fetch_canary_metrics(namespace, name))
                applied = _apply_plan(plan) if not plan.discovered else {}
                metrics.RECONCILE_TOTAL.labels(resource="inferenceservice", result="success").inc()
                log.info(
                    "reconciled name=%s actions=%s ready=%s",
                    name,
                    plan.summary()["actions"],
                    plan.status.is_ready,
                )
                return {
                    "applied": applied,
                    "summary": plan.summary(),
                    "status": plan.status.to_dict(),
                    "events": plan.events,
                    "durationSeconds": round(time.monotonic() - started, 4),
                }
            except Exception as exc:
                metrics.RECONCILE_TOTAL.labels(resource="inferenceservice", result="error").inc()
                log.exception("reconcile_failed name=%s err=%s", name, exc)
                raise kopf.TemporaryError(f"调和失败，将重试：{exc}", delay=15) from exc

    @kopf.timer(API_GROUP, API_VERSION, PLURAL, interval=60.0, initial_delay=30.0)
    async def periodic_analysis(
        spec: dict[str, Any], meta: dict[str, Any], body: dict[str, Any], **_kwargs: Any
    ) -> dict[str, Any] | None:
        """周期性灰度分析：时间驱动的逻辑（每 60s 评估一次权重推进）。

        为什么必须是 timer 而不是只靠 watch 事件？
        因为"错误率是否超限"这类判断依赖**时间窗口内的统计量**，
        没有 CR 变更事件也会发生变化 —— 只监听事件会导致灰度永远停在初始权重。
        """
        if not (spec.get("canary") or {}).get("enabled"):
            return None

        plan = plan_reconcile(
            {"metadata": meta, "spec": spec, "status": body.get("status", {})},
            _list_owned(meta.get("namespace", "default"), meta.get("name", "")),
            canary_metrics=_fetch_canary_metrics(meta.get("namespace", "default"), meta.get("name", "")),
        )
        _apply_plan(plan)
        log.info(
            "canary_tick name=%s weight=%s state=%s",
            meta.get("name"),
            plan.status.canary_weight,
            plan.status.canary_state,
        )
        # 注意：kopf.Patch 是 dict 的子类，**不是异常**，必须 return 而不是 raise。
        # 用 raise 会在运行时抛 TypeError（"exceptions must derive from BaseException"），
        # 导致灰度分析每 60s 崩一次。
        return dict(kopf.Patch({"status": plan.status.to_dict()}))

    @kopf.on.field(API_GROUP, API_VERSION, PLURAL, field="metadata.deletionTimestamp")
    async def on_delete(**_kwargs: Any) -> None:
        """CR 删除：子资源靠 ownerReference 级联回收，这里只记日志。

        如果不用 ownerReference 而手写删除逻辑，很容易漏删（比如新增了一种资源类型忘记加），
        最终留下"没人管但仍在消耗 GPU"的 Pod。
        """
        log.info("InferenceService 删除，子资源将由 ownerReference 级联回收")


__all__ = [
    "register_handlers",
]
