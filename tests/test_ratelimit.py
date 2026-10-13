"""多租户限流测试：令牌桶补充、突发、双维度配额、租户隔离。"""

from __future__ import annotations

import asyncio

import pytest

from aetherserve.gateway.ratelimit import RateLimitExceeded, TenantRateLimiter


class TestTokenBucket:
    async def test_burst_then_reject(self) -> None:
        rl = TenantRateLimiter(default_rps=1.0, default_burst=3, enabled=True)
        for _ in range(3):
            await rl.acquire("t1")  # 突发容量内应全部通过
        with pytest.raises(RateLimitExceeded) as ei:
            await rl.acquire("t1")
        assert ei.value.retry_after_s > 0
        assert rl.rejected_total == 1

    async def test_refill_over_time(self) -> None:
        rl = TenantRateLimiter(default_rps=100.0, default_burst=1, enabled=True)
        await rl.acquire("t1")
        with pytest.raises(RateLimitExceeded):
            await rl.acquire("t1")
        await asyncio.sleep(0.03)  # 100 rps → 30ms 补充约 3 个令牌
        await rl.acquire("t1")

    async def test_tenants_isolated(self) -> None:
        rl = TenantRateLimiter(default_rps=1.0, default_burst=1, enabled=True)
        await rl.acquire("tenant-a")
        with pytest.raises(RateLimitExceeded):
            await rl.acquire("tenant-a")
        await rl.acquire("tenant-b")  # 其他租户不受影响

    async def test_per_principal_override(self) -> None:
        rl = TenantRateLimiter(default_rps=1.0, default_burst=1, enabled=True)
        await rl.acquire("vip", rps=1000.0, burst=100)
        for _ in range(50):
            await rl.acquire("vip", rps=1000.0, burst=100)

    async def test_disabled_limiter_always_passes(self) -> None:
        rl = TenantRateLimiter(default_rps=0.001, default_burst=1, enabled=False)
        for _ in range(100):
            await rl.acquire("t1")


class TestTokenDimension:
    async def test_large_token_cost_rejected_when_budget_exhausted(self) -> None:
        rl = TenantRateLimiter(default_rps=1.0, default_burst=10, tokens_per_request_budget=1000)
        # 先消耗 token 维度（每秒补充 1000 token）
        await rl.acquire("t1", cost_tokens=9000)
        with pytest.raises(RateLimitExceeded) as ei:
            await rl.acquire("t1", cost_tokens=9000)
        assert "token" in str(ei.value)

    async def test_request_count_dimension_still_enforced(self) -> None:
        rl = TenantRateLimiter(default_rps=1.0, default_burst=2)
        await rl.acquire("t1", cost_tokens=0)
        await rl.acquire("t1", cost_tokens=0)
        with pytest.raises(RateLimitExceeded):
            await rl.acquire("t1", cost_tokens=0)

    async def test_oversized_single_request_not_permanently_blocked(self) -> None:
        """单请求成本超过整桶容量时只记账不拒绝，否则超大请求永远无法通过。"""
        rl = TenantRateLimiter(default_rps=1.0, default_burst=2, tokens_per_request_budget=1000)
        await rl.acquire("t1", cost_tokens=100_000)


class TestObservability:
    async def test_snapshot(self) -> None:
        rl = TenantRateLimiter(default_rps=10.0, default_burst=10)
        await rl.acquire("t1")
        snap = rl.snapshot()
        assert "t1" in snap
        assert snap["t1"]["rps"] == 10.0
        assert 0.0 <= snap["t1"]["utilization"] <= 1.0

    async def test_wait_hint(self) -> None:
        rl = TenantRateLimiter(default_rps=1.0, default_burst=1)
        await rl.acquire("t1")
        hint = rl.wait_hint("t1")
        assert 0.0 < hint <= 5.0
        assert rl.wait_hint("unknown") == 1.0
