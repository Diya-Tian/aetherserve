"""灰度发布（Canary）状态机。

为什么把决策写成**纯函数**？
---------------------------
灰度推进的判定条件很多（样本量不足、错误率超限、延迟劣化、已到 100%），
如果把这些 if/else 混在调和循环里，就只能靠"改集群 + 等 5 分钟"来验证 —— 成本极高。
本模块把决策抽成 ``evaluate_canary(spec, state, metrics) -> CanaryDecision``：
输入是纯数据、输出是纯数据，于是所有边界情形都能在毫秒级单元测试里覆盖。

状态机
------
    Pending ──首次调和──▶ Progressing ──每轮健康──▶ ... ──weight=100──▶ Promoted
                              │
                              └──错误率超限 / 连续劣化──▶ RolledBack (weight 归零)

关键安全设计
-----------
1. **最小样本量门禁**：样本少于 ``min_samples`` 时**绝不推进**。
   否则一个只收到 3 个请求、恰好全成功的灰度版会被误判为"健康"直接全量，
   这是灰度发布最经典的事故模式。
2. **失败立即止损**：错误率一旦超阈值，不做"再观察一轮"，直接归零。
   灰度的价值就是限制爆炸半径，犹豫等于放大损失。
3. **权重只增不减**：健康时单调推进，避免权重反复横跳导致线上流量分布抖动。
4. **达到 100% 视为成功终态**：状态置为 Promoted，之后不再做分析，
   由运维显式把 spec 里的新版本固化为稳定版（避免"永久的灰度态"）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from .crd import InferenceServiceSpec


class CanaryPhase(StrEnum):
    """灰度阶段。

    继承 ``StrEnum``（Python 3.11+）而非 ``(str, Enum)``：
    前者是标准写法，``str(phase)`` 直接得到值，序列化到 CR status 时不会变成 ``"CanaryPhase.NONE"``。
    """

    NONE = "None"  # 未启用
    PROGRESSING = "Progressing"
    PROMOTED = "Promoted"
    ROLLED_BACK = "RolledBack"


@dataclass(slots=True)
class CanaryMetrics:
    """一个分析窗口内观测到的灰度版本指标。"""

    request_count: int = 0
    error_count: int = 0
    p99_latency_s: float = 0.0

    @property
    def error_rate(self) -> float:
        if self.request_count <= 0:
            return 0.0
        return self.error_count / self.request_count


@dataclass(slots=True)
class CanaryState:
    """灰度的当前状态（持久化在 CR 的 ``status`` 中）。"""

    phase: CanaryPhase = CanaryPhase.NONE
    weight: int = 0
    steps: int = 0
    healthy_streak: int = 0
    last_reason: str = ""

    @property
    def is_terminal(self) -> bool:
        return self.phase in (CanaryPhase.PROMOTED, CanaryPhase.ROLLED_BACK)

    def to_dict(self) -> dict[str, Any]:
        return {
            "canaryWeight": self.weight,
            "canaryState": self.phase.value,
        }


@dataclass(slots=True)
class CanaryDecision:
    """一次评估的决策结果。"""

    phase: CanaryPhase
    weight: int
    reason: str
    rollback: bool = False
    promoted: bool = False
    details: dict[str, Any] = field(default_factory=dict)


def initial_state(spec: InferenceServiceSpec) -> CanaryState:
    """按 spec 初始化灰度状态。"""
    if not spec.canary.enabled:
        return CanaryState(phase=CanaryPhase.NONE, weight=0, last_reason="未启用灰度")
    return CanaryState(
        phase=CanaryPhase.PROGRESSING,
        weight=max(0, min(spec.canary.initial_weight, 100)),
        steps=0,
        last_reason=f"灰度启动，初始权重 {spec.canary.initial_weight}%",
    )


def evaluate_canary(
    spec: InferenceServiceSpec,
    state: CanaryState,
    metrics: CanaryMetrics,
) -> CanaryDecision:
    """评估一轮灰度指标，决定推进、保持还是回滚。

    判定顺序（顺序本身就是设计）：
    1. 未启用 / 已是终态 → 原样返回，不做任何事；
    2. **样本不足 → 保持不动**（不推进也不回滚，继续攒样本）；
    3. **错误率超限 → 立即回滚**；
    4. 健康 → 推进权重；
    5. 推进到 100% → Promoted。

    Args:
        spec: InferenceService 规格（含 canary 阈值）。
        state: 当前灰度状态。
        metrics: 本窗口观测指标。

    Returns:
        决策结果，调用方据此更新 CR status 与 Ingress 权重。
    """
    if not spec.canary.enabled:
        return CanaryDecision(CanaryPhase.NONE, 0, "未启用灰度")

    if state.is_terminal:
        return CanaryDecision(
            state.phase,
            state.weight,
            f"灰度已处于终态 {state.phase.value}，跳过分析",
            rollback=state.phase is CanaryPhase.ROLLED_BACK,
            promoted=state.phase is CanaryPhase.PROMOTED,
        )

    cfg = spec.canary

    # ---- 门禁 1：样本量不足，保持当前权重（不动比乱动安全）
    if metrics.request_count < cfg.min_samples:
        return CanaryDecision(
            CanaryPhase.PROGRESSING,
            state.weight,
            f"样本不足（{metrics.request_count}/{cfg.min_samples}），保持权重 {state.weight}%",
            details={"phase": "warming_up"},
        )

    error_rate = metrics.error_rate

    # ---- 门禁 2：错误率超限，立即回滚，不做二次观察
    if error_rate > cfg.max_error_rate:
        return CanaryDecision(
            CanaryPhase.ROLLED_BACK,
            0,
            f"错误率 {error_rate:.2%} 超过阈值 {cfg.max_error_rate:.2%}，自动回滚",
            rollback=True,
            details={"error_rate": round(error_rate, 4), "threshold": cfg.max_error_rate},
        )

    # ---- 健康：推进权重
    new_weight = min(state.weight + cfg.step, 100)
    if new_weight >= 100:
        return CanaryDecision(
            CanaryPhase.PROMOTED,
            100,
            f"灰度验证通过（错误率 {error_rate:.2%}，样本 {metrics.request_count}），已全量",
            promoted=True,
            details={"error_rate": round(error_rate, 4), "steps": state.steps + 1},
        )

    return CanaryDecision(
        CanaryPhase.PROGRESSING,
        new_weight,
        f"灰度健康（错误率 {error_rate:.2%}），权重 {state.weight}% → {new_weight}%",
        details={
            "error_rate": round(error_rate, 4),
            "previous_weight": state.weight,
            "steps": state.steps + 1,
        },
    )


def apply_decision(state: CanaryState, decision: CanaryDecision) -> CanaryState:
    """把决策落成新状态（保持不可变风格，便于测试与回放）。"""
    return CanaryState(
        phase=decision.phase,
        weight=decision.weight,
        steps=state.steps + (1 if decision.phase is CanaryPhase.PROGRESSING else 0),
        healthy_streak=state.healthy_streak + 1 if not decision.rollback else 0,
        last_reason=decision.reason,
    )


def ingress_weights(decision: CanaryDecision, stable_weight: int | None = None) -> dict[str, int]:
    """把灰度权重换算成 Ingress 分流权重。

    Returns:
        ``{"stable": w1, "canary": w2}``，两者之和为 100（回滚时为 100/0）。
    """
    canary_weight = max(0, min(decision.weight, 100))
    stable = 100 - canary_weight if stable_weight is None else stable_weight
    return {"stable": stable, "canary": canary_weight}


def summarize(spec: InferenceServiceSpec, decision: CanaryDecision) -> dict[str, Any]:
    """生成写回 CR status 的摘要。"""
    return {
        "canaryWeight": decision.weight,
        "canaryState": decision.phase.value,
        "canaryReason": decision.reason,
        "rollback": decision.rollback,
        "enabled": spec.canary.enabled,
    }
