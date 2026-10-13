"""批次构建策略。

批处理的目标：**在"等待更多请求以摊薄单次前向开销"与"不显著增加首 token 延迟"之间取平衡。**

三个约束旋钮
------------
1. ``max_batch_size``：单批最多几个请求 —— 受限于显存与 attention 计算量；
2. ``max_batch_tokens``：单批 token 预算 —— 直接决定 KV Cache 占用峰值；
3. ``max_wait_ms``：批构建最长等待 —— **这是延迟与吞吐的显式权衡点**：
   调大 → 批更大、吞吐更高、但 TTFT 变差；调小 → 延迟低、吞吐下降。
   生产上通常按模型大小与 SLA 设定（我们默认 20ms）。

长度分桶
--------
同一批内 prompt 长度差异过大会造成 padding 浪费：若批内最长 prompt 为 2048 而最短为 64，
则短请求要 padding 到 2048 参与计算，算力利用率被严重稀释。
``bucket_of()`` 把 prompt 长度按 ``bucket_step`` 归档，调度时倾向于优先聚合同桶请求，
用很小的实现代价换取可观的算力利用率提升。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class BatchPlan:
    """批构建约束（不可变，可安全跨协程共享）。"""

    max_batch_size: int = 16
    max_batch_tokens: int = 8192
    max_wait_ms: float = 20.0
    bucket_step: int = 128

    def __post_init__(self) -> None:
        if self.max_batch_size < 1:
            raise ValueError("max_batch_size 必须 >= 1")
        if self.max_batch_tokens < 1:
            raise ValueError("max_batch_tokens 必须 >= 1")
        if self.bucket_step < 1:
            raise ValueError("bucket_step 必须 >= 1")

    @classmethod
    def from_settings(cls, s: object) -> BatchPlan:
        """从 ``config.BatchingSettings`` 构造。"""
        return cls(
            max_batch_size=getattr(s, "max_batch_size", 16),
            max_batch_tokens=getattr(s, "max_batch_tokens", 8192),
            max_wait_ms=getattr(s, "max_wait_ms", 20.0),
            bucket_step=getattr(s, "bucket_step", 128),
        )


def bucket_of(prompt_tokens: int, step: int = 128) -> int:
    """把 prompt 长度映射到桶索引（向上取整，保证大 prompt 不落入小桶）。"""
    return (max(prompt_tokens, 1) + step - 1) // step


def build_batches(
    items: list[object],
    plan: BatchPlan,
    tokens_of: Callable[[object], int] | None = None,
) -> list[list[object]]:
    """把一批请求按 token 预算与长度分桶切分为多个子批。

    这是 ``AdmissionQueue`` 之外的第二层批构建：单元测试与离线重放会直接调用它，
    因此保持为纯函数（无副作用、无 I/O），便于用属性测试验证不变量。

    Args:
        items: 待分批的请求对象列表（任意带 token 估算的对象）。
        plan: 批约束。
        tokens_of: ``(item) -> int`` 的取值函数；默认读取 ``est_total_tokens`` 属性。

    Returns:
        子批列表，满足：每个子批 token 总量（除单请求超预算情况外）不超过 ``max_batch_tokens``，
        且子批数量不超过 ``max_batch_size``。
    """
    if tokens_of is None:

        def _default_tokens_of(item: object) -> int:
            return int(getattr(item, "est_total_tokens", 1))

        tokens_of = _default_tokens_of

    # 按桶排序，同桶相邻 → 同批 padding 更少
    step = plan.bucket_step
    ordered = sorted(
        items,
        key=lambda it: (bucket_of(tokens_of(it), step), tokens_of(it)),
    )

    batches: list[list[object]] = []
    current: list[object] = []
    current_tokens = 0

    for item in ordered:
        t = tokens_of(item)
        over_size = len(current) >= plan.max_batch_size
        over_tokens = current and (current_tokens + t > plan.max_batch_tokens)
        if over_size or over_tokens:
            batches.append(current)
            current = []
            current_tokens = 0
        current.append(item)
        current_tokens += t

    if current:
        batches.append(current)
    return batches
