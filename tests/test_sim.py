"""仿真与压测工具的测试。

为什么必须测「仿真」？
--------------------
性能报告中的加速比数字全部源自这套仿真。若仿真实现有 bug，
数字就不可信 —— 评审时会被问穿。因此这里验证的是**模型的性质**，而不只是"能跑"：
- 时间模型必须满足"批越大单步成本越高、但摊到每请求越低"（单调性）；
- 批同步必须让同批请求**共享**一次解码开销（不能各付一次）；
- 串行对照必须真的按"每请求一次前向"计费（否则对比无意义）；
- 负载驱动必须产出合理分位数与 100% 成功率。
"""

from __future__ import annotations

import asyncio

import pytest

from aetherserve.config import Settings
from aetherserve.sim import (
    LatencyStats,
    SerialReferenceBackend,
    SimulatedBackend,
    SimulatedHardware,
    make_prompt,
    reference_hardware,
    run_load_test,
)


class TestSimulatedHardware:
    def test_prefill_linear_in_tokens(self) -> None:
        hw = SimulatedHardware(prefill_ms_per_token=0.5)
        assert hw.prefill_ms(100) == pytest.approx(50.0)
        assert hw.prefill_ms(1000) == pytest.approx(500.0)
        assert hw.prefill_ms(200) == pytest.approx(2 * hw.prefill_ms(100))

    def test_prefill_non_zero_for_empty(self) -> None:
        """空 prompt 也要有基础开销（模型前向不可能零成本）。"""
        hw = SimulatedHardware(prefill_ms_per_token=0.5)
        assert hw.prefill_ms(0) > 0

    def test_weight_share_decreases_with_batch(self) -> None:
        hw = reference_hardware()
        shares = [hw.weight_share(b) for b in (1, 2, 4, 8, 16, 32, 64)]
        assert shares[0] == pytest.approx(1.0), "批大小为 1 时应付全部权重读取成本"
        assert shares == sorted(shares, reverse=True), f"摊薄系数必须单调递减: {shares}"
        assert all(s > 0 for s in shares), "权重成本不可能为零"

    def test_decode_step_increases_with_batch(self) -> None:
        """单步成本随批增大而增大（attention 计算量上升）。"""
        hw = reference_hardware()
        steps = [hw.decode_step_ms(b) for b in (1, 2, 4, 8, 16, 32)]
        assert steps == sorted(steps), f"单步成本必须单调递增: {steps}"

    def test_decode_cost_per_request_decreases(self) -> None:
        """核心性质：摊到每个请求的解码成本随批增大而下降 —— 批处理收益的来源。"""
        hw = reference_hardware()
        per_req = [hw.decode_step_ms(b) / b for b in (1, 2, 4, 8, 16, 32)]
        assert per_req == sorted(per_req, reverse=True), f"每请求成本必须单调递减: {per_req}"
        assert per_req[-1] < per_req[0] / 3, "批 32 时的每请求成本应显著低于批 1"

    def test_serial_step_is_batch_size_times_single(self) -> None:
        """串行等价成本 = 每请求各跑一次前向，因此随批大小线性增长。"""
        hw = reference_hardware()
        assert hw.serial_step_ms(1) == pytest.approx(hw.decode_step_ms(1))
        assert hw.serial_step_ms(8) == pytest.approx(hw.decode_step_ms(1) * 8)

    def test_theoretical_ratio_matches_definition(self) -> None:
        hw = reference_hardware()
        for b in (1, 2, 8, 16, 32):
            assert hw.theoretical_ratio(b) == pytest.approx(hw.serial_step_ms(b) / hw.decode_step_ms(b))
        assert hw.theoretical_ratio(1) == pytest.approx(1.0)
        assert hw.theoretical_ratio(16) > hw.theoretical_ratio(8) > 1.0

    def test_theoretical_total_linear_in_requests(self) -> None:
        hw = reference_hardware()
        one = hw.theoretical_total_ms(8, 8, 256, 32)
        two = hw.theoretical_total_ms(8, 16, 256, 32)
        assert two == pytest.approx(one * 2), "请求数翻倍、批大小不变时总耗时应翻倍"


class TestBatchRendezvous:
    """批同步：同批请求必须共享一次解码开销。

    测试策略说明（为什么不再用「并发 N 个协程然后断言批大小」）
    --------------------------------------------------------
    依赖事件循环调度顺序的断言在 Windows 上是**不可靠**的：
    即使加 ``asyncio.Barrier``，协程释放顺序也不保证同 tick 全部就绪，
    实测会出现"部分请求落到下一个批窗口"从而批大小偏离预期。
    因此这里分两层验证：
    - **确定性层**：直接驱动 ``_BatchRendezvous``，验证计费与去重逻辑（无调度竞态）；
    - **鲁棒性层**：跑真实后端，只断言**稳健性质**（步数、批大小上界、计费随批单调下降）。
    """

    async def test_rendezvous_bills_once_for_shared_step(self) -> None:
        """确定性验证：N 个等待者只应产生 1 次计费、1 个融合步。"""
        from aetherserve.sim.backend import _BatchRendezvous

        rv = _BatchRendezvous()
        size = 6
        barrier = asyncio.Barrier(size)
        per_step = 10.0

        async def waiter(i: int) -> int:
            async with barrier:
                pass
            return await rv.step(lambda b: per_step)

        results = await asyncio.gather(*(waiter(i) for i in range(size)))

        assert rv.steps_completed == 1, f"共享同一步应只记 1 步，实际 {rv.steps_completed}"
        assert rv.billing_ms == pytest.approx(per_step), (
            f"同一次融合步只能计费一次，实际 {rv.billing_ms}（若为 {per_step * size} 说明按等待者重复计费）"
        )
        assert results == [size] * size, f"每个等待者都应看到本步批大小 {size}，实际 {results}"

    async def test_rendezvous_accumulates_across_steps(self) -> None:
        """确定性验证：多步依次累加，每步各计一次。"""
        from aetherserve.sim.backend import _BatchRendezvous

        rv = _BatchRendezvous()
        size = 3
        steps = 4
        barrier = asyncio.Barrier(size)

        async def waiter(_i: int) -> None:
            async with barrier:
                pass
            for _ in range(steps):
                await rv.step(lambda b: float(b))

        await asyncio.gather(*(waiter(i) for i in range(size)))
        assert rv.steps_completed == steps, f"应有 {steps} 步，实际 {rv.steps_completed}"
        assert rv.billing_ms == pytest.approx(float(size) * steps)

    async def test_billing_per_request_settles_as_batch_grows(self) -> None:
        """批越大，摊到每个请求的解码计费越低（在真实后端上验证，但只断言单调性）。

        这里基于**计费值**而非墙钟：计费是纯公式累加，不受定时器精度影响。
        """
        from aetherserve.backends.base import GenerateParams

        hw = SimulatedHardware(prefill_ms_per_token=0.0, decode_base_ms=9.0, decode_per_request_ms=0.55)

        async def run(n: int) -> float:
            backend = SimulatedBackend(hardware=hw)
            barrier = asyncio.Barrier(n)

            async def one(i: int) -> None:
                async with barrier:
                    pass
                async for _ in backend.generate(f"r{i}", make_prompt(32), GenerateParams(max_tokens=2)):
                    pass

            await asyncio.gather(*(one(i) for i in range(n)))
            assert backend.num_decode_steps == 2, (
                f"每个请求 2 步，融合后应为 2 步，实际 {backend.num_decode_steps}"
            )
            return backend.billing_ms / n

        per_req_1 = await run(1)
        per_req_4 = await run(4)
        per_req_16 = await run(16)
        assert per_req_16 < per_req_4 < per_req_1, (
            f"每请求计费应随批增大而下降：1→{per_req_1:.3f} 4→{per_req_4:.3f} 16→{per_req_16:.3f}"
        )

    async def test_batch_size_never_exceeds_concurrency(self) -> None:
        """鲁棒性：任何一步的批大小都不会超过并发请求数。"""
        from aetherserve.backends.base import GenerateParams

        hw = SimulatedHardware(prefill_ms_per_token=0.0, decode_base_ms=2.0, decode_per_request_ms=0.1)
        backend = SimulatedBackend(hardware=hw)
        n = 4
        barrier = asyncio.Barrier(n)

        async def one(i: int) -> None:
            async with barrier:
                pass
            async for _ in backend.generate(f"r{i}", make_prompt(32), GenerateParams(max_tokens=3)):
                pass

        await asyncio.gather(*(one(i) for i in range(n)))
        assert backend.step_batch_sizes, "应至少记录一个解码步"
        assert max(backend.step_batch_sizes) <= n, f"批大小超过并发数: {backend.step_batch_sizes}"
        assert backend.num_decode_steps >= 3, "至少应有 3 个融合步"

    async def test_single_request_batch_size_one(self) -> None:
        backend = SimulatedBackend(hardware=SimulatedHardware(prefill_ms_per_token=0.0))
        from aetherserve.backends.base import GenerateParams

        async for _ in backend.generate("solo", make_prompt(32), GenerateParams(max_tokens=3)):
            pass
        assert set(backend.step_batch_sizes) == {1}
        assert backend.num_decode_steps == 3

    async def test_prefill_is_concurrent_not_summed(self) -> None:
        """批内 prefill 并发发生，因此整批只付"单请求 prefill 时长"。

        这是校验脚本口径正确的关键：若按"每请求各自计时再求和"，会虚高 N 倍。
        """
        hw = SimulatedHardware(prefill_ms_per_token=1.0, decode_base_ms=0.0, decode_per_request_ms=0.0)
        backend = SimulatedBackend(hardware=hw)
        msgs = make_prompt(100)  # → 约 100 token → 约 100ms prefill
        from aetherserve.backends.base import GenerateParams

        prompt_tokens = backend.estimate_prompt_tokens(msgs)

        async def one(i: int) -> None:
            async for _ in backend.generate(f"r{i}", msgs, GenerateParams(max_tokens=1)):
                pass

        import time

        t0 = time.perf_counter()
        await asyncio.gather(*(one(i) for i in range(8)))
        elapsed_ms = (time.perf_counter() - t0) * 1000

        per_request_prefill = hw.prefill_ms(prompt_tokens)
        # 若 8 个 prefill 串行相加，耗时会接近 8 × per_request；并发则接近 1 × per_request
        assert elapsed_ms < per_request_prefill * 3, (
            f"prefill 未并发执行：8 请求耗时 {elapsed_ms:.1f}ms，单请求 prefill {per_request_prefill:.1f}ms"
        )

    async def test_batch_windows_consistent(self) -> None:
        backend = SimulatedBackend(hardware=SimulatedHardware(prefill_ms_per_token=0.0))
        from aetherserve.backends.base import GenerateParams

        async def one(i: int) -> None:
            async for _ in backend.generate(f"r{i}", make_prompt(32), GenerateParams(max_tokens=2)):
                pass

        await asyncio.gather(*(one(i) for i in range(4)))
        windows = backend.batch_windows()
        assert len(windows) >= 1
        for w in windows.values():
            assert w["batch_size"] >= 1
            assert w["prompt_tokens"] > 0

    async def test_theoretical_prefill_counts_windows_not_requests(self) -> None:
        """解析 prefill 应按批窗口计数，而不是按请求累加。"""
        hw = SimulatedHardware(prefill_ms_per_token=0.5, decode_base_ms=0.5, decode_per_request_ms=0.1)
        backend = SimulatedBackend(hardware=hw)
        from aetherserve.backends.base import GenerateParams

        async def one(i: int) -> None:
            async for _ in backend.generate(f"r{i}", make_prompt(64), GenerateParams(max_tokens=2)):
                pass

        await asyncio.gather(*(one(i) for i in range(8)))
        windows = backend.batch_windows()
        expected = sum(hw.prefill_ms(w["prompt_tokens"]) for w in windows.values())
        assert backend.theoretical_prefill_ms() == pytest.approx(expected)
        # 每个窗口的请求数之和应等于 8
        assert sum(w["batch_size"] for w in windows.values()) == 8


class TestSerialReference:
    """串行对照必须真的按"每请求一次前向"计费 —— 否则整个加速比对比都无意义。

    与 ``TestBatchRendezvous`` 同样的策略：断言**计费模型**（纯公式累加），
    不依赖墙钟，因为 Windows 定时器粒度（≈15ms）会让毫秒级断言抖动。
    """

    async def test_serial_step_billing_equals_n_times_single_forward(self) -> None:
        hw = SimulatedHardware(prefill_ms_per_token=0.0, decode_base_ms=4.0, decode_per_request_ms=1.0)
        # 串行成本定义 = 每请求各跑一次前向 → B × decode_step_ms(1)
        assert hw.serial_step_ms(1) == pytest.approx(hw.decode_step_ms(1))
        assert hw.serial_step_ms(8) == pytest.approx(hw.decode_step_ms(1) * 8)
        # 且严格大于融合步成本（这正是批处理收益的来源）
        assert hw.serial_step_ms(8) > hw.decode_step_ms(8)

    async def test_serial_backend_uses_serial_billing(self) -> None:
        """真实串行后端：无论几个并发请求共享一步，计费都按 N × 单请求前向。"""
        from aetherserve.backends.base import GenerateParams

        hw = SimulatedHardware(prefill_ms_per_token=0.0, decode_base_ms=4.0, decode_per_request_ms=1.0)
        backend = SerialReferenceBackend(hardware=hw)
        n = 4
        barrier = asyncio.Barrier(n)

        async def one(i: int) -> None:
            async with barrier:
                pass
            async for _ in backend.generate(f"r{i}", make_prompt(32), GenerateParams(max_tokens=2)):
                pass

        await asyncio.gather(*(one(i) for i in range(n)))

        assert backend.num_decode_steps == 2, f"融合步应为 2，实际 {backend.num_decode_steps}"
        # 计费 = Σ 每步 serial_step_ms(该步批大小)，且严格大于同样步数的融合计费
        batched = SimulatedBackend(hardware=hw)
        barrier2 = asyncio.Barrier(n)

        async def one2(i: int) -> None:
            async with barrier2:
                pass
            async for _ in batched.generate(f"r{i}", make_prompt(32), GenerateParams(max_tokens=2)):
                pass

        await asyncio.gather(*(one2(i) for i in range(n)))

        assert backend.billing_ms > batched.billing_ms, (
            f"串行计费 {backend.billing_ms:.2f}ms 应高于批处理计费 {batched.billing_ms:.2f}ms"
        )
        # 每个请求的成本：串行 = 单请求前向；批处理 = 融合步摊到 B 个请求
        serial_per_req = backend.billing_ms / 2 / n
        batched_per_req = batched.billing_ms / 2 / n
        assert serial_per_req == pytest.approx(hw.decode_step_ms(1)), (
            f"串行每请求成本应等于单请求前向成本，实际 {serial_per_req:.3f}"
        )
        assert batched_per_req < serial_per_req

    async def test_serial_slower_than_batched(self) -> None:
        """同条件下串行必须比批处理慢（这是整个性能主张的基础）。"""
        hw = SimulatedHardware(prefill_ms_per_token=0.2, decode_base_ms=9.0, decode_per_request_ms=0.55)
        from aetherserve.backends.base import GenerateParams

        async def run(backend: SimulatedBackend) -> float:
            import time

            async def one(i: int) -> None:
                async for _ in backend.generate(f"r{i}", make_prompt(128), GenerateParams(max_tokens=8)):
                    pass

            t0 = time.perf_counter()
            await asyncio.gather(*(one(i) for i in range(8)))
            return (time.perf_counter() - t0) * 1000

        serial_ms = await run(SerialReferenceBackend(hardware=hw))
        batched_ms = await run(SimulatedBackend(hardware=hw))
        assert batched_ms < serial_ms, f"批处理({batched_ms:.1f}ms) 应快于串行({serial_ms:.1f}ms)"
        assert serial_ms / batched_ms > 2.0, "加速比应显著（>2×）"


class TestGpuUtilization:
    async def test_utilization_rises_with_batch(self) -> None:
        hw = SimulatedHardware(prefill_ms_per_token=0.0, decode_base_ms=9.0, decode_per_request_ms=0.55)
        from aetherserve.backends.base import GenerateParams

        async def run(n: int) -> float:
            backend = SimulatedBackend(hardware=hw)

            async def one(i: int) -> None:
                async for _ in backend.generate(f"r{i}", make_prompt(32), GenerateParams(max_tokens=2)):
                    pass

            await asyncio.gather(*(one(i) for i in range(n)))
            return backend.estimated_gpu_utilization()

        small = await run(1)
        large = await run(16)
        assert large > small, f"批越大利用率应越高: {small:.3f} → {large:.3f}"
        assert 0.0 <= small <= 1.0
        assert 0.0 <= large <= 1.0

    async def test_utilization_zero_without_batches(self) -> None:
        assert SimulatedBackend().estimated_gpu_utilization() == 0.0


class TestLatencyStats:
    def test_empty(self) -> None:
        s = LatencyStats.from_samples([])
        assert s.count == 0
        assert s.p50_ms == 0.0

    def test_single_sample(self) -> None:
        s = LatencyStats.from_samples([5.0])
        assert s.count == 1
        assert s.p50_ms == s.p99_ms == 5.0
        assert s.stddev_ms == 0.0

    def test_percentiles_monotonic(self) -> None:
        samples = [float(i) for i in range(1, 101)]
        s = LatencyStats.from_samples(samples)
        assert s.min_ms == 1.0
        assert s.max_ms == 100.0
        assert s.p50_ms <= s.p95_ms <= s.p99_ms
        assert s.p50_ms == pytest.approx(50.0, abs=1.0)
        assert s.p99_ms == pytest.approx(99.0, abs=1.0)

    def test_to_dict_keys(self) -> None:
        d = LatencyStats.from_samples([1.0, 2.0]).to_dict()
        assert set(d) == {"count", "mean_ms", "p50_ms", "p95_ms", "p99_ms", "min_ms", "max_ms", "stddev_ms"}

    def test_mean_and_stddev(self) -> None:
        s = LatencyStats.from_samples([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        assert s.mean_ms == pytest.approx(5.0)
        assert s.stddev_ms == pytest.approx(2.0)


class TestMakePrompt:
    def test_prompt_length_roughly_target(self) -> None:
        """make_prompt 用于控制 prompt token 数，误差不应过大（影响实验可比性）。"""
        hw = reference_hardware()
        backend = SimulatedBackend(hardware=hw)
        for target in (64, 128, 256):
            tokens = backend.estimate_prompt_tokens(make_prompt(target))
            assert 0.7 * target <= tokens <= 1.6 * target, f"target={target} 实际={tokens}"

    def test_returns_single_user_message(self) -> None:
        msgs = make_prompt(32)
        assert len(msgs) == 1
        assert msgs[0].role == "user"
        assert isinstance(msgs[0].content, str) and msgs[0].content


class TestRunLoadTest:
    """端到端驱动：验证压测工具本身产出合理结果。"""

    def _settings(self, **kw: object) -> Settings:
        s = Settings()
        s.environment = "test"
        s.batching.max_queue_size = 512
        s.batching.max_queue_tokens = 1_000_000
        s.backend.model = "sim-model"
        for k, v in kw.items():
            setattr(s.batching, k, v)
        return s

    async def test_load_test_completes_for_both_modes(self) -> None:
        """压测工具本身必须能跑完两种模式并产出完整结果。

        注意：这里**不**断言"批处理 QPS 更高"。
        原因：QPS 是墙钟量，受 Windows 定时器粒度与事件循环开销影响，
        在单测环境里做阈值断言会不稳定（本项目实测多次抖动）。
        "批处理更快"这一主张由 ``TestSimulatedHardware`` 与 ``TestSerialReference``
        在**计费模型层面**确定性验证，那里不受调度与定时器影响。
        真实加速比证据来自 ``bench/run_bench.py`` 与 ``bench/verify_model.py``（独立自校验）。
        """
        hw = SimulatedHardware(prefill_ms_per_token=0.1, decode_base_ms=16.0, decode_per_request_ms=0.4)

        batched = await run_load_test(
            SimulatedBackend(hardware=hw),
            self._settings(max_batch_size=8, worker_concurrency=1),
            label="batched",
            concurrency=8,
            requests=16,
            max_tokens=4,
            prompt_tokens=64,
            warmup=False,
        )
        serial = await run_load_test(
            SerialReferenceBackend(hardware=hw),
            self._settings(max_batch_size=1, max_wait_ms=0.0, worker_concurrency=1),
            label="serial",
            concurrency=8,
            requests=16,
            max_tokens=4,
            prompt_tokens=64,
            warmup=False,
        )

        for r in (batched, serial):
            assert r.completed == 16, f"{r.label} 未全部完成: {r.completed}"
            assert r.success_rate == 1.0
            assert r.wall_time_s > 0
            assert r.qps > 0
            assert r.ttft.count == 16
            assert r.e2e.count == 16
            assert r.ttft.p50_ms > 0
            assert r.ttft.p99_ms >= r.ttft.p50_ms
            assert r.e2e.p99_ms >= r.e2e.p50_ms

        # 结构性差异（不依赖时间）：批处理确实聚合了请求，串行没有
        assert batched.observed_peak_batch > 1, "批处理应观测到批聚合"
        assert serial.observed_peak_batch == 1, "串行不应聚合"
        assert batched.engine_stats.get("peak_batch_size", 0) > 1

    async def test_result_fields_populated(self) -> None:
        hw = SimulatedHardware(prefill_ms_per_token=0.05, decode_base_ms=2.0, decode_per_request_ms=0.2)
        result = await run_load_test(
            SimulatedBackend(hardware=hw),
            self._settings(max_batch_size=4, worker_concurrency=1),
            label="t",
            concurrency=4,
            requests=8,
            max_tokens=3,
            prompt_tokens=32,
            warmup=False,
        )
        assert result.wall_time_s > 0
        assert result.qps > 0
        assert result.tokens_per_second > 0
        assert result.ttft.count == 8
        assert result.e2e.count == 8
        assert result.ttft.p50_ms > 0
        d = result.to_dict()
        assert d["label"] == "t"
        assert "ttft" in d and "tpot" in d and "e2e" in d

    async def test_no_rejections_within_capacity(self) -> None:
        hw = SimulatedHardware(prefill_ms_per_token=0.02, decode_base_ms=1.0, decode_per_request_ms=0.1)
        result = await run_load_test(
            SimulatedBackend(hardware=hw),
            self._settings(max_batch_size=4, worker_concurrency=1, max_queue_size=64),
            label="cap",
            concurrency=4,
            requests=12,
            max_tokens=2,
            prompt_tokens=32,
            warmup=False,
        )
        assert result.rejected == 0
        assert result.completed == 12

    async def test_warmup_not_counted(self) -> None:
        """预热请求不应进入统计（否则会污染首请求的开销）。"""
        hw = SimulatedHardware(prefill_ms_per_token=0.02, decode_base_ms=1.0, decode_per_request_ms=0.1)
        result = await run_load_test(
            SimulatedBackend(hardware=hw),
            self._settings(max_batch_size=2, worker_concurrency=1),
            label="warm",
            concurrency=2,
            requests=6,
            max_tokens=2,
            prompt_tokens=32,
            warmup=True,
        )
        assert result.completed == 6, f"预热被计入统计: {result.completed}"
        assert result.ttft.count == 6
