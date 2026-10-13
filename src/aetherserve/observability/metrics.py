"""Prometheus 指标定义（LLM 推理专项）。

为什么指标要分这么细？
----------------------
传统 Web 服务的 RED 指标（Rate/Error/Duration）无法定位推理服务的性能瓶颈：
一次推理请求的延迟由 **排队 → 预填充(prefill) → 逐 token 解码(decode)** 三段构成，
其中 prefill 主要吃算力、decode 主要吃显存带宽。
只有把 TTFT（首 token 时间）、TPOT（相邻 token 间隔）、批大小、KV Cache 水位分开观测，
才能在面试里回答"**你的瓶颈在哪一段，怎么证明**"。
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

# 独立 registry：便于在测试中隔离，也便于 /metrics 只暴露本服务指标
REGISTRY = CollectorRegistry()

# 延迟桶：覆盖本地小模型的典型区间（10ms ~ 30s）
_LATENCY_BUCKETS = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.075,
    0.1,
    0.15,
    0.2,
    0.3,
    0.5,
    0.75,
    1.0,
    1.5,
    2.0,
    3.0,
    5.0,
    8.0,
    13.0,
    21.0,
    30.0,
)

# ---------------------------------------------------------------- 请求维度
REQUESTS_TOTAL = Counter(
    "aetherserve_requests_total",
    "推理请求总数",
    ["model", "tenant", "status"],
    registry=REGISTRY,
)

REQUEST_LATENCY = Histogram(
    "aetherserve_request_latency_seconds",
    "端到端请求延迟（含排队）",
    ["model", "stream"],
    buckets=_LATENCY_BUCKETS,
    registry=REGISTRY,
)

# ---------------------------------------------------------------- 引擎维度
QUEUE_DEPTH = Gauge(
    "aetherserve_queue_depth",
    "当前在途请求数（已准入未完成）",
    registry=REGISTRY,
)

QUEUE_WAIT = Histogram(
    "aetherserve_queue_wait_seconds",
    "请求在队列中的等待时间（批构建延迟的主要来源）",
    buckets=(0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
    registry=REGISTRY,
)

BATCH_SIZE = Histogram(
    "aetherserve_batch_size",
    "每次批执行的请求数（观察批处理是否真的聚合了请求）",
    buckets=(1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64),
    registry=REGISTRY,
)

BATCH_TOKENS = Histogram(
    "aetherserve_batch_tokens",
    "每次批执行的 token 预算占用",
    buckets=(128, 512, 1024, 2048, 4096, 8192, 16384),
    registry=REGISTRY,
)

INFLIGHT_TOKENS = Gauge(
    "aetherserve_inflight_tokens",
    "在途请求预估 KV Cache token 总量（显存预算使用量）",
    registry=REGISTRY,
)

PREEMPTIONS_TOTAL = Counter(
    "aetherserve_preemptions_total",
    "因显存预算不足被拒绝（背压）的请求数",
    ["reason"],
    registry=REGISTRY,
)

# ---------------------------------------------------------------- 推理质量维度
TTFT = Histogram(
    "aetherserve_ttft_seconds",
    "首 token 时间 Time-To-First-Token（用户感知延迟的核心指标）",
    ["model"],
    buckets=_LATENCY_BUCKETS,
    registry=REGISTRY,
)

TPOT = Histogram(
    "aetherserve_tpot_seconds",
    "相邻 token 输出间隔 Time-Per-Output-Token（解码速度）",
    ["model"],
    buckets=(0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2, 0.4, 1.0),
    registry=REGISTRY,
)

TOKENS_TOTAL = Counter(
    "aetherserve_tokens_total",
    "token 总数",
    ["model", "tenant", "kind"],  # kind: prompt | completion
    registry=REGISTRY,
)

# ---------------------------------------------------------------- 后端维度
BACKEND_ERRORS = Counter(
    "aetherserve_backend_errors_total",
    "后端调用错误数",
    ["backend", "kind"],
    registry=REGISTRY,
)

CIRCUIT_STATE = Gauge(
    "aetherserve_circuit_breaker_state",
    "熔断器状态：0=closed 1=half_open 2=open",
    ["backend"],
    registry=REGISTRY,
)

# ---------------------------------------------------------------- 多租户维度
RATELIMIT_REJECTED = Counter(
    "aetherserve_ratelimit_rejected_total",
    "被限流拒绝的请求数",
    ["tenant"],
    registry=REGISTRY,
)

# ---------------------------------------------------------------- 算子/控制面
RECONCILE_TOTAL = Counter(
    "aetherserve_reconcile_total",
    "Operator 调和次数",
    ["resource", "result"],
    registry=REGISTRY,
)

RECONCILE_DURATION = Histogram(
    "aetherserve_reconcile_duration_seconds",
    "Operator 单次调和耗时",
    ["resource"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0),
    registry=REGISTRY,
)


def render_metrics() -> bytes:
    """序列化为 Prometheus 文本格式，供 ``/metrics`` 端点使用。"""
    from prometheus_client import generate_latest

    return generate_latest(REGISTRY)
