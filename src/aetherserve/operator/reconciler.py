"""调和（Reconcile）核心逻辑 —— 不依赖真实集群，可完整单元测试。

Kubernetes 控制器的标准模型
--------------------------
    「观测实际状态」 → 「计算期望状态」 → 「求差」 → 「收敛差异」

本模块实现中间的**求差与决策**部分，并把"如何写回集群"交给 ``handlers.py``（kopf）。
这样拆分带来两个直接好处：
1. 调和逻辑的全部边界（资源缺失、spec 变更、灰度推进、CR 删除清理）
   都能在毫秒级单元测试里验证，不需要起集群；
2. 换掉 kopf（例如改用 Java Operator SDK 或 Go controller-runtime）时，
   决策逻辑可以原样复用。

幂等性的具体实现
----------------
- 每个子资源名由 ``CR 名 + 角色`` 唯一确定（不含随机数、时间戳）；
- Deployment 的 Pod 模板带 ``spec-hash`` annotation，用于判断"是否真的需要更新"；
- 调和函数**不读也不写**本地状态：给定同样的输入必然给出同样的输出。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .builders import existing_spec_hash, manifest_identity
from .canary import CanaryMetrics, CanaryPhase, evaluate_canary
from .crd import (
    InferenceServiceStatus,
    parse_spec,
    validate_spec,
)


@dataclass(slots=True)
class ReconcileAction:
    """单个子资源需要执行的动作。"""

    op: str  # "create" | "update" | "noop"
    manifest: dict[str, Any]
    reason: str = ""

    @property
    def identity(self) -> tuple[str, str, str]:
        return manifest_identity(self.manifest)

    def __str__(self) -> str:
        _api_version, kind, name = self.identity
        return f"{self.op}:{kind}/{name}({self.reason})"


@dataclass(slots=True)
class ReconcilePlan:
    """一次调和的完整计划。"""

    actions: list[ReconcileAction] = field(default_factory=list)
    status: InferenceServiceStatus = field(default_factory=InferenceServiceStatus)
    errors: list[str] = field(default_factory=list)
    events: list[str] = field(default_factory=list)

    @property
    def discovered(self) -> bool:
        """CR 被标记删除（或校验失败）时，调和应停止创建新资源。"""
        return bool(self.errors)

    def summary(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for a in self.actions:
            counts[a.op] = counts.get(a.op, 0) + 1
        return {
            "actions": counts,
            "errors": self.errors,
            "ready": self.status.is_ready,
            "endpoint": self.status.endpoint,
            "canary_state": self.status.canary_state,
            "canary_weight": self.status.canary_weight,
        }


def _find_existing(
    kind: str,
    name: str,
    existing: list[dict[str, Any]],
) -> dict[str, Any] | None:
    for res in existing:
        if res.get("kind") == kind and res.get("metadata", {}).get("name") == name:
            return res
    return None


def plan_reconcile(
    cr: dict[str, Any],
    existing: list[dict[str, Any]] | None = None,
    canary_metrics: CanaryMetrics | None = None,
) -> ReconcilePlan:
    """计算一次调和的计划。

    Args:
        cr: Custom Resource 的完整对象（含 metadata / spec / status）。
        existing: 当前集群中已存在的、由本 CR 拥有的子资源列表。
        canary_metrics: 灰度分析窗口的观测指标（由 Prometheus 查询得到）。

    Returns:
        ``ReconcilePlan``：要 apply 的资源清单 + 要写回的状态。
    """
    from . import builders as b

    plan = ReconcilePlan()
    existing = existing or []

    metadata = cr.get("metadata", {})
    name = str(metadata.get("name", ""))
    namespace = str(metadata.get("namespace", "default"))
    generation = int(metadata.get("generation", 0) or 0)

    if not name:
        plan.errors.append("CR 缺少 metadata.name")
        return plan

    # ---------------------------------------------------------- 1. 解析与校验
    spec = parse_spec(cr.get("spec", {}) or {})
    errors = validate_spec(spec)
    if errors:
        plan.errors.extend(errors)
        plan.status.observed_generation = generation
        plan.status.canary_state = CanaryPhase.NONE.value
        return plan

    # ---------------------------------------------------------- 2. 灰度决策
    prev_status = cr.get("status", {}) or {}
    canary_decision = None
    if spec.canary.enabled:
        from .canary import CanaryState, initial_state

        prev_phase = str(prev_status.get("canaryState", CanaryPhase.NONE.value))
        try:
            phase = CanaryPhase(prev_phase)
        except ValueError:
            phase = CanaryPhase.NONE

        if phase is CanaryPhase.NONE:
            state = initial_state(spec)
        else:
            state = CanaryState(
                phase=phase,
                weight=int(prev_status.get("canaryWeight", 0) or 0),
                last_reason=str(prev_status.get("canaryReason", "")),
            )

        canary_decision = evaluate_canary(spec, state, canary_metrics or CanaryMetrics())
        plan.events.append(f"canary: {canary_decision.reason}")

    # ---------------------------------------------------------- 3. 计算期望状态
    include_canary = bool(
        spec.canary.enabled
        and canary_decision is not None
        and not canary_decision.rollback
        and not canary_decision.promoted
    )
    desired = b.build_all(name, namespace, spec, cr, include_canary=include_canary)

    # ---------------------------------------------------------- 4. 求差
    desired_ids = {manifest_identity(m) for m in desired}
    for manifest in desired:
        _, kind, res_name = manifest_identity(manifest)
        current = _find_existing(kind, res_name, existing)

        if current is None:
            plan.actions.append(ReconcileAction("create", manifest, reason="资源不存在"))
            continue

        # 变更检测：Deployment 比对 spec hash，其他资源比对完整 desried spec
        if kind == "Deployment":
            want = existing_spec_hash(manifest)
            have = existing_spec_hash(current)
            if want != have:
                plan.actions.append(ReconcileAction("update", manifest, reason="spec hash 变化"))
            else:
                plan.actions.append(ReconcileAction("noop", manifest, reason="spec 未变化"))
        else:
            if _spec_changed(current, manifest):
                plan.actions.append(ReconcileAction("update", manifest, reason="spec 变化"))
            else:
                plan.actions.append(ReconcileAction("noop", manifest, reason="spec 未变化"))

    # ---------------------------------------------------------- 5. 清理孤儿资源
    # 例如灰度回滚后，canary 的 Deployment/Service 不再出现在期望集合里，必须删除
    for res in existing:
        kind = str(res.get("kind", ""))
        res_name = str(res.get("metadata", {}).get("name", ""))
        if kind not in {"Deployment", "Service", "HorizontalPodAutoscaler", "PodDisruptionBudget"}:
            continue
        if (str(res.get("apiVersion", "")), kind, res_name) not in desired_ids:
            # 只清理我们自己管理的资源，避免误删他人资源（安全检查）
            labels = res.get("metadata", {}).get("labels", {}) or {}
            if labels.get("app.kubernetes.io/managed-by") == b.MANAGED_BY:
                plan.actions.append(ReconcileAction("delete", res, reason="不再属于期望状态（如灰度回滚）"))

    # ---------------------------------------------------------- 6. 组装状态
    status = plan.status
    status.observed_generation = generation
    status.replicas = spec.replicas
    ready_count = _count_ready(existing)
    status.ready_replicas = ready_count
    status.endpoint = f"http://{name}-stable.{namespace}.svc.cluster.local"

    if canary_decision is not None:
        status.canary_weight = canary_decision.weight
        status.canary_state = canary_decision.phase.value
    else:
        status.canary_weight = 0
        status.canary_state = CanaryPhase.NONE.value

    from .crd import Condition

    if not errors:
        status.set_condition(
            Condition(
                type="Ready",
                status="True" if ready_count >= max(spec.replicas, 1) else "False",
                reason="Reconciled" if ready_count >= max(spec.replicas, 1) else "Progressing",
                message=f"{ready_count}/{spec.replicas} 副本就绪",
            )
        )
        status.set_condition(
            Condition(
                type="Progressing",
                status="True" if ready_count < max(spec.replicas, 1) else "False",
                reason="Progressing" if ready_count < max(spec.replicas, 1) else "Complete",
            )
        )
        if canary_decision is not None and canary_decision.phase in (
            CanaryPhase.PROGRESSING,
            CanaryPhase.PROMOTED,
        ):
            status.set_condition(
                Condition(
                    type="CanaryRunning",
                    status="True" if canary_decision.phase is CanaryPhase.PROGRESSING else "False",
                    reason=canary_decision.phase.value,
                    message=canary_decision.reason,
                )
            )
        if canary_decision is not None and canary_decision.rollback:
            status.set_condition(
                Condition(
                    type="CanaryFailed",
                    status="True",
                    reason="ErrorRateExceeded",
                    message=canary_decision.reason,
                )
            )
            status.set_condition(
                Condition(
                    type="Ready", status="True", reason="RolledBack", message="灰度已回滚，稳定版继续服务"
                )
            )

    return plan


def _spec_changed(current: dict[str, Any], desired: dict[str, Any]) -> bool:
    """比较两个资源的关键字段是否需要更新。

    只比较 ``spec``：``metadata`` 里的 ``resourceVersion`` 等字段
    由 API Server 维护，逐字段全量比较会产生大量无意义的写入。
    """
    return _normalize(current.get("spec", {})) != _normalize(desired.get("spec", {}))


def _normalize(value: Any) -> Any:
    """把结构归一化，消除"键顺序不同但语义相同"造成的假变更。"""
    if isinstance(value, dict):
        return {k: _normalize(v) for k, v in sorted(value.items()) if k != "resourceVersion"}
    if isinstance(value, list):
        return [_normalize(v) for v in value]
    return value


def _count_ready(existing: list[dict[str, Any]]) -> int:
    """从实际状态中统计就绪副本数。"""
    for res in existing:
        if (
            res.get("kind") == "Deployment"
            and res.get("metadata", {}).get("labels", {}).get("aetherserve.io/role", "stable") == "stable"
        ):
            return int(res.get("status", {}).get("readyReplicas", 0) or 0)
    return 0


def build_initial_status() -> InferenceServiceStatus:
    """新建 CR 时的初始状态。"""
    from .crd import Condition

    status = InferenceServiceStatus()
    status.set_condition(Condition(type="Ready", status="False", reason="Initializing"))
    status.set_condition(Condition(type="Progressing", status="True", reason="Reconciling"))
    return status


__all__ = [
    "ReconcileAction",
    "ReconcilePlan",
    "build_initial_status",
    "plan_reconcile",
]
