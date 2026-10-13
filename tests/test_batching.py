"""批构建策略测试：验证不变量而非具体实现细节。"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from aetherserve.engine.batching import BatchPlan, bucket_of, build_batches


@dataclass
class FakeItem:
    est_total_tokens: int
    prompt_tokens: int = 0
    name: str = ""


class TestBucketOf:
    def test_rounds_up(self) -> None:
        assert bucket_of(1, 128) == 1
        assert bucket_of(128, 128) == 1
        assert bucket_of(129, 128) == 2
        assert bucket_of(256, 128) == 2
        assert bucket_of(0, 128) == 1  # 空 prompt 归入最小桶

    def test_monotonic(self) -> None:
        prev = 0
        for tokens in range(1, 2000, 37):
            cur = bucket_of(tokens, 128)
            assert cur >= prev
            prev = cur


class TestBuildBatches:
    def test_respects_max_batch_size(self) -> None:
        plan = BatchPlan(max_batch_size=3, max_batch_tokens=10_000, bucket_step=64)
        items = [FakeItem(est_total_tokens=10) for _ in range(10)]
        batches = build_batches(items, plan)
        assert len(batches) == 4
        assert all(len(b) <= 3 for b in batches)
        assert sum(len(b) for b in batches) == 10

    def test_respects_token_budget(self) -> None:
        plan = BatchPlan(max_batch_size=100, max_batch_tokens=100, bucket_step=64)
        items = [FakeItem(est_total_tokens=40) for _ in range(5)]
        batches = build_batches(items, plan)
        # 每批最多 2 个（40+40=80，再加一个会超 100）
        assert all(sum(i.est_total_tokens for i in b) <= 100 + 40 for b in batches)
        assert [len(b) for b in batches] == [2, 2, 1]

    def test_single_oversized_item_still_dispatched(self) -> None:
        """单个请求超过批预算时不能丢弃 —— 否则大请求会被永久饿死。"""
        plan = BatchPlan(max_batch_size=8, max_batch_tokens=64, bucket_step=64)
        items = [FakeItem(est_total_tokens=5000)]
        batches = build_batches(items, plan)
        assert len(batches) == 1
        assert batches[0][0].est_total_tokens == 5000

    def test_length_bucketing_groups_similar_prompts(self) -> None:
        """同批内 prompt 长度应接近，以减少 padding 浪费。"""
        plan = BatchPlan(max_batch_size=2, max_batch_tokens=10_000, bucket_step=128)
        items = [
            FakeItem(est_total_tokens=100, name="short-a"),
            FakeItem(est_total_tokens=2000, name="long-a"),
            FakeItem(est_total_tokens=110, name="short-b"),
        ]
        batches = build_batches(items, plan)
        # 两个短请求应被分到同一批
        first = {i.name for i in batches[0]}
        assert first == {"short-a", "short-b"}, f"分桶失败: {batches}"

    def test_all_items_preserved(self) -> None:
        plan = BatchPlan(max_batch_size=3, max_batch_tokens=200, bucket_step=32)
        items = [FakeItem(est_total_tokens=(i % 7) * 10 + 1) for i in range(50)]
        batches = build_batches(items, plan)
        flat = [it for b in batches for it in b]
        assert len(flat) == 50
        assert {id(x) for x in flat} == {id(x) for x in items}

    def test_empty_input(self) -> None:
        assert build_batches([], BatchPlan()) == []

    def test_invalid_plan_rejected(self) -> None:
        with pytest.raises(ValueError):
            BatchPlan(max_batch_size=0)
        with pytest.raises(ValueError):
            BatchPlan(max_batch_tokens=0)
        with pytest.raises(ValueError):
            BatchPlan(bucket_step=0)

    def test_plan_from_settings(self) -> None:
        class S:
            max_batch_size = 7
            max_batch_tokens = 999
            max_wait_ms = 12.5
            bucket_step = 32

        p = BatchPlan.from_settings(S())
        assert (p.max_batch_size, p.max_batch_tokens, p.max_wait_ms, p.bucket_step) == (7, 999, 12.5, 32)
