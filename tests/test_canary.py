"""灰度发布状态机测试。

这些用例覆盖了灰度发布最容易出事的边界：
样本不足、错误率刚好卡阈值、权重推进到 100%、回滚后不再推进。
如果不写这些测试，验证方式就只剩"改集群等 5 分钟看结果"，成本高到没人会测。
"""

from __future__ import annotations

import pytest

from aetherserve.operator.canary import (
    CanaryMetrics,
    CanaryPhase,
    CanaryState,
    apply_decision,
    evaluate_canary,
    ingress_weights,
    initial_state,
    summarize,
)
from aetherserve.operator.crd import parse_spec


def spec_with_canary(**overrides: object):
    canary = {
        "enabled": True,
        "initialWeight": 10,
        "step": 20,
        "maxErrorRate": 0.05,
        "minSamples": 100,
    }
    canary.update(overrides)
    return parse_spec({"model": "m", "replicas": 2, "canary": canary})


class TestInitialState:
    def test_disabled(self) -> None:
        spec = parse_spec({"model": "m"})
        state = initial_state(spec)
        assert state.phase is CanaryPhase.NONE
        assert state.weight == 0

    def test_enabled_starts_progressing(self) -> None:
        state = initial_state(spec_with_canary())
        assert state.phase is CanaryPhase.PROGRESSING
        assert state.weight == 10

    def test_weight_clamped(self) -> None:
        state = initial_state(spec_with_canary(initialWeight=150))
        assert state.weight == 100


class TestSampleGate:
    """样本量门禁：不满足就绝不推进 —— 这是灰度最经典的事故来源。"""

    def test_insufficient_samples_holds_weight(self) -> None:
        spec = spec_with_canary(minSamples=100)
        state = CanaryState(phase=CanaryPhase.PROGRESSING, weight=10)
        decision = evaluate_canary(spec, state, CanaryMetrics(request_count=3, error_count=0))
        assert decision.phase is CanaryPhase.PROGRESSING
        assert decision.weight == 10, "样本不足时权重必须保持不动"
        assert not decision.promoted
        assert "样本不足" in decision.reason

    def test_insufficient_samples_is_not_rollback(self) -> None:
        """样本不足也不能回滚 —— 3 个请求里恰好 1 个失败不构成 33% 错误率的证据。"""
        spec = spec_with_canary(minSamples=100, maxErrorRate=0.05)
        state = CanaryState(phase=CanaryPhase.PROGRESSING, weight=10)
        decision = evaluate_canary(spec, state, CanaryMetrics(request_count=3, error_count=1))
        assert not decision.rollback
        assert decision.weight == 10

    def test_zero_requests_holds(self) -> None:
        spec = spec_with_canary(minSamples=10)
        state = CanaryState(phase=CanaryPhase.PROGRESSING, weight=20)
        decision = evaluate_canary(spec, state, CanaryMetrics())
        assert decision.weight == 20
        assert decision.phase is CanaryPhase.PROGRESSING


class TestProgression:
    def test_healthy_advances_by_step(self) -> None:
        spec = spec_with_canary(step=20)
        state = CanaryState(phase=CanaryPhase.PROGRESSING, weight=10)
        decision = evaluate_canary(spec, state, CanaryMetrics(request_count=200, error_count=2))
        assert decision.weight == 30
        assert decision.phase is CanaryPhase.PROGRESSING
        assert not decision.promoted

    def test_reaches_100_promotes(self) -> None:
        spec = spec_with_canary(step=20)
        state = CanaryState(phase=CanaryPhase.PROGRESSING, weight=90)
        decision = evaluate_canary(spec, state, CanaryMetrics(request_count=500, error_count=0))
        assert decision.weight == 100
        assert decision.phase is CanaryPhase.PROMOTED
        assert decision.promoted

    def test_step_overshoot_clamps_to_100(self) -> None:
        spec = spec_with_canary(step=50)
        state = CanaryState(phase=CanaryPhase.PROGRESSING, weight=80)
        decision = evaluate_canary(spec, state, CanaryMetrics(request_count=500))
        assert decision.weight == 100
        assert decision.phase is CanaryPhase.PROMOTED

    def test_error_rate_exactly_at_threshold_is_healthy(self) -> None:
        """边界语义：``>`` 而非 ``>=`` —— 恰好等于阈值不算失败，避免抖动误回滚。"""
        spec = spec_with_canary(maxErrorRate=0.05, step=10)
        state = CanaryState(phase=CanaryPhase.PROGRESSING, weight=10)
        decision = evaluate_canary(spec, state, CanaryMetrics(request_count=1000, error_count=50))
        assert not decision.rollback
        assert decision.weight == 20

    def test_weight_monotonic_increase(self) -> None:
        spec = spec_with_canary(step=15)
        weight = 10
        for _ in range(5):
            state = CanaryState(phase=CanaryPhase.PROGRESSING, weight=weight)
            decision = evaluate_canary(spec, state, CanaryMetrics(request_count=200, error_count=0))
            assert decision.weight >= weight, "权重绝不能回退（会造成流量分布抖动）"
            weight = decision.weight


class TestRollback:
    def test_high_error_rate_rolls_back(self) -> None:
        spec = spec_with_canary(maxErrorRate=0.05)
        state = CanaryState(phase=CanaryPhase.PROGRESSING, weight=50)
        decision = evaluate_canary(spec, state, CanaryMetrics(request_count=1000, error_count=100))
        assert decision.rollback
        assert decision.phase is CanaryPhase.ROLLED_BACK
        assert decision.weight == 0, "回滚必须把权重归零"
        assert "回滚" in decision.reason

    def test_rollback_is_immediate_no_second_chance(self) -> None:
        """错误率超限不做二次观察：灰度的价值就是限制爆炸半径。"""
        spec = spec_with_canary(maxErrorRate=0.01)
        state = CanaryState(phase=CanaryPhase.PROGRESSING, weight=10)
        decision = evaluate_canary(spec, state, CanaryMetrics(request_count=1000, error_count=11))
        assert decision.rollback

    def test_rolled_back_state_is_terminal(self) -> None:
        spec = spec_with_canary()
        state = CanaryState(phase=CanaryPhase.ROLLED_BACK, weight=0)
        decision = evaluate_canary(spec, state, CanaryMetrics(request_count=10_000, error_count=0))
        assert decision.phase is CanaryPhase.ROLLED_BACK
        assert decision.weight == 0
        assert "终态" in decision.reason

    def test_promoted_state_is_terminal(self) -> None:
        spec = spec_with_canary()
        state = CanaryState(phase=CanaryPhase.PROMOTED, weight=100)
        decision = evaluate_canary(spec, state, CanaryMetrics(request_count=10_000, error_count=5000))
        assert decision.phase is CanaryPhase.PROMOTED
        assert decision.weight == 100, "已全量的灰度不应因后续指标被回滚"


class TestDisabled:
    def test_disabled_spec_never_progresses(self) -> None:
        spec = parse_spec({"model": "m"})
        decision = evaluate_canary(spec, CanaryState(), CanaryMetrics(request_count=10_000))
        assert decision.phase is CanaryPhase.NONE
        assert decision.weight == 0


class TestStateTransition:
    def test_apply_decision_updates_state(self) -> None:
        spec = spec_with_canary(step=20)
        state = initial_state(spec)
        decision = evaluate_canary(spec, state, CanaryMetrics(request_count=200))
        new_state = apply_decision(state, decision)
        assert new_state.weight == 30
        assert new_state.steps == 1
        assert new_state.phase is CanaryPhase.PROGRESSING
        assert new_state.last_reason == decision.reason

    def test_apply_decision_resets_streak_on_rollback(self) -> None:
        spec = spec_with_canary(maxErrorRate=0.01)
        state = CanaryState(phase=CanaryPhase.PROGRESSING, weight=50, healthy_streak=7)
        decision = evaluate_canary(spec, state, CanaryMetrics(request_count=1000, error_count=100))
        new_state = apply_decision(state, decision)
        assert new_state.healthy_streak == 0
        assert new_state.phase is CanaryPhase.ROLLED_BACK

    def test_state_dict_shape(self) -> None:
        state = CanaryState(phase=CanaryPhase.PROGRESSING, weight=30)
        d = state.to_dict()
        assert d == {"canaryWeight": 30, "canaryState": "Progressing"}


class TestIngressWeights:
    def test_full_canary(self) -> None:
        from aetherserve.operator.canary import CanaryDecision

        w = ingress_weights(CanaryDecision(CanaryPhase.PROMOTED, 100, "done"))
        assert w == {"stable": 0, "canary": 100}

    def test_rollback_all_to_stable(self) -> None:
        from aetherserve.operator.canary import CanaryDecision

        w = ingress_weights(CanaryDecision(CanaryPhase.ROLLED_BACK, 0, "rollback"))
        assert w == {"stable": 100, "canary": 0}

    def test_partial_split_sums_to_100(self) -> None:
        from aetherserve.operator.canary import CanaryDecision

        w = ingress_weights(CanaryDecision(CanaryPhase.PROGRESSING, 30, "step"))
        assert w["stable"] + w["canary"] == 100
        assert w["canary"] == 30

    def test_clamped(self) -> None:
        from aetherserve.operator.canary import CanaryDecision

        w = ingress_weights(CanaryDecision(CanaryPhase.PROGRESSING, 250, "over"))
        assert w["canary"] == 100


class TestSummarize:
    def test_summary_fields(self) -> None:
        spec = spec_with_canary()
        state = initial_state(spec)
        decision = evaluate_canary(spec, state, CanaryMetrics(request_count=200))
        s = summarize(spec, decision)
        assert s["canaryWeight"] == 30
        assert s["canaryState"] == "Progressing"
        assert s["enabled"] is True
        assert isinstance(s["canaryReason"], str)


class TestErrorRateComputation:
    def test_zero_requests(self) -> None:
        assert CanaryMetrics().error_rate == 0.0

    def test_all_failed(self) -> None:
        assert CanaryMetrics(request_count=10, error_count=10).error_rate == 1.0

    def test_partial(self) -> None:
        assert CanaryMetrics(request_count=200, error_count=10).error_rate == pytest.approx(0.05)
