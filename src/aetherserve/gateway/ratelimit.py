"""多租户限流：令牌桶 + 每租户独立配额。

选型理由
--------
- **令牌桶 vs 漏桶 vs 固定窗口**：
  固定窗口在窗口切换处会出现"双倍突发"；
  漏桶强制恒定速率、无法应对合法突发流量；
  令牌桶允许在 ``burst`` 范围内突发、长期速率被 ``rps`` 约束，
  最契合推理服务"平时低频、偶发批量请求"的真实流量形态。

- **为什么令牌数按 token 计费也要限？**
  单请求成本差异极大（10 token 和 4096 token 相差两个数量级），
  只按请求数限流会让大请求打爆 GPU。
  因此实现"请求数 + 预估 token 数"双维度配额（token 维度在 ``check_tokens``）。

实现细节：使用单调时钟 ``time.monotonic()`` 而非墙钟，
避免 NTP 校时回拨导致令牌凭空增加。
"""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass, field

from ..observability import metrics
from ..observability.logging import get_logger

log = get_logger(__name__)


class RateLimitExceeded(Exception):  # noqa: N818 - 该名称是既有公开 API，改名会破坏调用方
    """触发限流（429）。

    命名说明：Ruff 的 N818 建议异常类以 ``Error`` 结尾。
    这里保留 ``RateLimitExceeded`` 是刻意的 —— 它是网关对外的契约名，
    且语义上描述的是"**状态**被超出"而非"发生了一个错误"。
    """

    def __init__(self, message: str, retry_after_s: float = 1.0) -> None:
        super().__init__(message)
        self.retry_after_s = retry_after_s


@dataclass
class _Bucket:
    """单个租户的令牌桶状态。"""

    rps: float
    burst: int
    tokens: float = field(default=0.0)
    token_tokens: float = field(default=0.0)  # 第二维度：token 配额
    last_refill: float = field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
        if self.tokens == 0.0:
            self.tokens = float(self.burst)
        if self.token_tokens == 0.0:
            self.token_tokens = float(self.burst) * 1000.0


class TenantRateLimiter:
    """按租户维度的双令牌桶限流器。"""

    def __init__(
        self,
        default_rps: float = 20.0,
        default_burst: int = 40,
        tokens_per_request_budget: int = 1000,
        enabled: bool = True,
    ) -> None:
        self._default_rps = default_rps
        self._default_burst = default_burst
        self._token_budget = max(tokens_per_request_budget, 1)
        self._enabled = enabled
        self._buckets: dict[str, _Bucket] = {}
        self._lock = asyncio.Lock()
        self.rejected_total = 0

    # ------------------------------------------------------------ 内部
    def _bucket_for(self, tenant: str, rps: float | None, burst: int | None) -> _Bucket:
        bucket = self._buckets.get(tenant)
        if bucket is None:
            bucket = _Bucket(rps=rps or self._default_rps, burst=burst or self._default_burst)
            self._buckets[tenant] = bucket
        elif rps is not None:
            bucket.rps = rps
            bucket.burst = max(burst or bucket.burst, 1)
        return bucket

    @staticmethod
    def _refill(bucket: _Bucket, now: float) -> None:
        elapsed = max(now - bucket.last_refill, 0.0)
        bucket.tokens = min(float(bucket.burst), bucket.tokens + elapsed * bucket.rps)
        # token 维度：每秒预算 = rps * 单请求预算
        cap = float(bucket.burst) * 1000.0
        bucket.token_tokens = min(cap, bucket.token_tokens + elapsed * bucket.rps * 1000.0)
        bucket.last_refill = now

    # ------------------------------------------------------------ 对外接口
    async def acquire(
        self,
        tenant: str,
        cost_tokens: int = 0,
        rps: float | None = None,
        burst: int | None = None,
    ) -> None:
        """尝试获取一次请求配额。

        Args:
            tenant: 租户标识。
            cost_tokens: 本次请求的预估 token 成本（prompt+completion）。
            rps: 覆盖默认速率。
            burst: 覆盖默认突发容量。

        Raises:
            RateLimitExceeded: 配额不足（携带建议的重试等待时间）。
        """
        if not self._enabled:
            return

        async with self._lock:
            bucket = self._bucket_for(tenant, rps, burst)
            now = time.monotonic()
            self._refill(bucket, now)

            if bucket.tokens < 1.0:
                need = 1.0 - bucket.tokens
                retry_after = need / max(bucket.rps, 1e-6)
                self.rejected_total += 1
                metrics.RATELIMIT_REJECTED.labels(tenant=tenant).inc()
                raise RateLimitExceeded(
                    f"租户 {tenant} 请求速率超限（{bucket.rps:.0f} req/s）",
                    retry_after_s=min(max(retry_after, 0.05), 5.0),
                )

            # token 维度：单请求成本超过整桶容量时只做记账不拒绝，避免大请求永久无法通过
            token_cost = float(max(cost_tokens, 0))
            if (
                token_cost
                and bucket.token_tokens < token_cost
                and token_cost <= self._token_budget * bucket.burst
            ):
                deficit = token_cost - bucket.token_tokens
                retry_after = deficit / max(bucket.rps * 1000.0, 1e-6)
                self.rejected_total += 1
                metrics.RATELIMIT_REJECTED.labels(tenant=tenant).inc()
                raise RateLimitExceeded(
                    f"租户 {tenant} token 配额不足（需要 {int(token_cost)}）",
                    retry_after_s=min(max(retry_after, 0.05), 5.0),
                )

            bucket.tokens -= 1.0
            bucket.token_tokens = max(0.0, bucket.token_tokens - token_cost)

    def snapshot(self) -> dict[str, dict[str, float]]:
        """当前各租户桶状态（供调试与 Grafana 展示）。"""
        now = time.monotonic()
        out: dict[str, dict[str, float]] = {}
        for tenant, bucket in self._buckets.items():
            self._refill(bucket, now)
            out[tenant] = {
                "rps": bucket.rps,
                "burst": float(bucket.burst),
                "tokens": round(bucket.tokens, 3),
                "utilization": round(1.0 - bucket.tokens / max(bucket.burst, 1), 4),
            }
        return out

    def wait_hint(self, tenant: str) -> float:
        """给 429 响应计算 Retry-After（秒）。"""
        bucket = self._buckets.get(tenant)
        if bucket is None:
            return 1.0
        self._refill(bucket, time.monotonic())
        if bucket.tokens >= 1.0:
            return 0.0
        return min(math.ceil((1.0 - bucket.tokens) / max(bucket.rps, 1e-6) * 20) / 20, 5.0)
