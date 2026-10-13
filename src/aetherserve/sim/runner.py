"""负载测试驱动：在进程内直接驱动引擎，测量吞吐与延迟分位数。

为什么不用 locust 跑 HTTP？
--------------------------
两种方式各有用途，本项目两者都做：
- **本驱动（进程内）**：绕开 HTTP 与序列化，单独隔离出"调度引擎"的性能，
  噪声小、可确定性复现，用于生成对照实验数据（开/关批处理）；
- **locust（HTTP 端到端）**：测包含网络、协议解析、SSE 编码在内的真实用户体验。

先做进程内，是因为它能干净地回答"批处理带来多少收益"这个核心问题。
"""

from __future__ import annotations

import asyncio
import math
import statistics
import time
from dataclasses import dataclass, field
from typing import Any

from ..backends.base import GenerateParams, InferenceBackend
from ..config import Settings
from ..engine import build_engine
from ..gateway.schemas import ChatMessage


@dataclass(slots=True)
class LatencyStats:
    """延迟统计（毫秒）。"""

    count: int = 0
    mean_ms: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    stddev_ms: float = 0.0

    @classmethod
    def from_samples(cls, samples: list[float]) -> LatencyStats:
        if not samples:
            return cls()
        ordered = sorted(samples)
        n = len(ordered)

        def pct(p: float) -> float:
            if n == 1:
                return ordered[0]
            idx = min(max(math.ceil(p * n) - 1, 0), n - 1)
            return ordered[idx]

        return cls(
            count=n,
            mean_ms=round(statistics.fmean(ordered), 3),
            p50_ms=round(pct(0.50), 3),
            p95_ms=round(pct(0.95), 3),
            p99_ms=round(pct(0.99), 3),
            min_ms=round(ordered[0], 3),
            max_ms=round(ordered[-1], 3),
            stddev_ms=round(statistics.pstdev(ordered), 3) if n > 1 else 0.0,
        )

    def to_dict(self) -> dict[str, float | int]:
        return {
            "count": self.count,
            "mean_ms": self.mean_ms,
            "p50_ms": self.p50_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
            "stddev_ms": self.stddev_ms,
        }


@dataclass(slots=True)
class LoadTestResult:
    """一次压测的完整结果。"""

    label: str
    concurrency: int
    requests: int
    max_tokens: int
    prompt_tokens: int
    wall_time_s: float = 0.0
    completed: int = 0
    failed: int = 0
    rejected: int = 0
    ttft: LatencyStats = field(default_factory=LatencyStats)
    tpot: LatencyStats = field(default_factory=LatencyStats)
    e2e: LatencyStats = field(default_factory=LatencyStats)
    observed_peak_batch: int = 0
    mean_batch_size: float = 0.0
    gpu_utilization: float = 0.0
    engine_stats: dict[str, Any] = field(default_factory=dict)

    @property
    def qps(self) -> float:
        return round(self.completed / self.wall_time_s, 3) if self.wall_time_s > 0 else 0.0

    @property
    def tokens_per_second(self) -> float:
        total = self.completed * self.max_tokens
        return round(total / self.wall_time_s, 2) if self.wall_time_s > 0 else 0.0

    @property
    def success_rate(self) -> float:
        attempted = self.completed + self.failed + self.rejected
        return round(self.completed / attempted, 4) if attempted else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "concurrency": self.concurrency,
            "requests": self.requests,
            "max_tokens": self.max_tokens,
            "prompt_tokens": self.prompt_tokens,
            "wall_time_s": round(self.wall_time_s, 4),
            "completed": self.completed,
            "failed": self.failed,
            "rejected": self.rejected,
            "qps": self.qps,
            "tokens_per_second": self.tokens_per_second,
            "success_rate": self.success_rate,
            "observed_peak_batch": self.observed_peak_batch,
            "mean_batch_size": round(self.mean_batch_size, 3),
            "gpu_utilization": round(self.gpu_utilization, 4),
            "ttft": self.ttft.to_dict(),
            "tpot": self.tpot.to_dict(),
            "e2e": self.e2e.to_dict(),
            "engine_stats": self.engine_stats,
        }


def make_prompt(target_tokens: int) -> list[ChatMessage]:
    """构造指定长度的中文 prompt（中文约 1 token/字，便于精确控制）。"""
    # 减去消息结构开销（estimate_prompt_tokens 会加 4 + 8）
    body = "这是一段用于压测的提示词内容" * (target_tokens // 14 + 1)
    return [ChatMessage(role="user", content=body[: max(target_tokens, 1)])]


async def run_load_test(
    backend: InferenceBackend,
    settings: Settings,
    *,
    label: str,
    concurrency: int = 16,
    requests: int = 64,
    max_tokens: int = 32,
    prompt_tokens: int = 128,
    warmup: bool = True,
) -> LoadTestResult:
    """驱动引擎完成一轮负载测试。

    实现要点：
    - 用"持续并发窗口"而非"一次性全部提交"，模拟真实到达过程：
      始终保持 ``concurrency`` 个请求在途，完成一个立刻补一个；
    - TTFT 从"提交时刻"开始计时（含排队），因为用户感知的就是这个；
    - TPOT 只统计相邻 token 之间的间隔（解码速度），不含首 token。
    """
    engine = build_engine(settings, backend)
    await engine.start()

    result = LoadTestResult(
        label=label,
        concurrency=concurrency,
        requests=requests,
        max_tokens=max_tokens,
        prompt_tokens=prompt_tokens,
    )
    ttft_samples: list[float] = []
    tpot_samples: list[float] = []
    e2e_samples: list[float] = []
    messages = make_prompt(prompt_tokens)

    async def one_request(index: int) -> None:
        t_submit = time.perf_counter()
        try:
            handle = await engine.submit(
                model=settings.backend.model,
                tenant="loadtest",
                messages=messages,
                params=GenerateParams(max_tokens=max_tokens, temperature=0.0),
                request_id=f"{label}-{index}",
            )
        except Exception:
            result.rejected += 1
            return

        first_token_at: float | None = None
        last_token_at: float | None = None
        try:
            async for chunk in handle.stream():
                if not chunk.text:
                    continue
                now = time.perf_counter()
                if first_token_at is None:
                    first_token_at = now
                    ttft_samples.append((now - t_submit) * 1000.0)
                else:
                    tpot_samples.append((now - last_token_at) * 1000.0)  # type: ignore[operator]
                last_token_at = now
            e2e_samples.append((time.perf_counter() - t_submit) * 1000.0)
            result.completed += 1
        except Exception:
            result.failed += 1

    try:
        if warmup:
            # 预热：让后端完成懒加载 / 连接建立，避免把首次开销算进结果
            warm = await engine.submit(
                settings.backend.model, "warmup", messages, GenerateParams(max_tokens=1), request_id="warmup"
            )
            async for _ in warm.stream():
                pass

        pending: set[asyncio.Task[None]] = set()
        next_index = 0
        t_start = time.perf_counter()

        while next_index < requests or pending:
            while next_index < requests and len(pending) < concurrency:
                task = asyncio.create_task(one_request(next_index))
                pending.add(task)
                next_index += 1
            if pending:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for d in done:
                    with_done = d.exception() if not d.cancelled() else None
                    if with_done is not None:
                        result.failed += 1

        result.wall_time_s = time.perf_counter() - t_start
    finally:
        snap = engine.stats_snapshot()
        await engine.stop(drain=False)

    result.ttft = LatencyStats.from_samples(ttft_samples)
    result.tpot = LatencyStats.from_samples(tpot_samples)
    result.e2e = LatencyStats.from_samples(e2e_samples)
    result.engine_stats = snap.get("scheduler", {})

    batches_seen = list(getattr(backend, "batches_seen", []) or [])
    if batches_seen:
        result.mean_batch_size = statistics.fmean(batches_seen)
    result.observed_peak_batch = int(getattr(backend, "peak_concurrency", 0) or 0)
    # 只有仿真后端才提供 GPU 利用率估算，用 getattr 做能力探测（duck typing）
    util_fn = getattr(backend, "estimated_gpu_utilization", None)
    if callable(util_fn):
        result.gpu_utilization = float(util_fn())

    return result
