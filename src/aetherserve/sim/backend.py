"""推理耗时仿真后端：把"连续批处理的收益"变成可复现、可验证的数字。

为什么需要它？
--------------
本项目的核心主张是"自研连续批处理引擎显著提升吞吐"。要证明这一点有两个办法：
1. 上真实 GPU 跑 vLLM —— 最理想，但依赖 Linux + CUDA + 大显存，且结果受驱动/型号影响；
2. **用物理量级正确的耗时模型仿真** —— 不依赖硬件，结果可复现、可解释、可自校验。

时间模型的物理依据（显存带宽受限 vs 算力受限）
---------------------------------------------
Transformer 推理的两个阶段瓶颈完全不同：

- **Prefill（预填充）**：把整个 prompt 送进模型算 attention 与 FFN。
  算术强度高，**受算力（FLOPs）限制** →
  处理共 ``P`` 个 prompt token 的时间 ≈ ``P × prefill_ms_per_token``（**线性**）。

- **Decode（逐 token 解码）**：每步只生成 1 个 token，却要把**全部模型权重**
  从显存读一遍，算术强度极低，**受显存带宽限制** →
  处理 1 步的时间 ≈ ``decode_base_ms × 权重读取摊薄系数 + decode_per_request_ms × B``。
  权重读取是**固定成本**，被批内所有请求摊薄 —— 这正是批处理收益的来源。

批同步（batch rendezvous）—— 本实现的关键
-----------------------------------------
朴素实现（每个请求独立计时）是**错的**：真实推理引擎的解码是
"把当前所有活跃请求融合成一次前向计算"，而不是"每个请求各跑一次前向"。
若按每请求独立计时，批内请求的耗时互不相干，批处理收益会被完全错误地估算。

本实现让同一个批内的所有请求在**同一个解码步上同步**：每个请求在生成第 i 个 token 时，
都会登记到一个按"步"聚合的屏障；屏障收齐批内所有请求后，只等待**一次**
``decode_step_ms(B)``，然后同时放行。这与 vLLM 的 step 循环语义一致，
也是 ``bench/verify_model.py`` 能通过理论校验的前提。

诚实的边界声明
--------------
本仿真的**绝对数值不代表真实 GPU 性能**；它保证的是**相对关系正确**：
"批处理相对串行的吞吐增益"与解析模型一致（偏差 < 15%，见 ``verify_model.py``）。
真实 GPU 数据由 ``bench/run_bench.py --mode vllm`` 采集后替换性能报告中的对应行。
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass

from ..backends.base import GenerateParams, InferenceBackend, TokenChunk
from ..gateway.schemas import ChatMessage

_MS = 1e-3


@dataclass(slots=True)
class SimulatedHardware:
    """仿真硬件的耗时系数（默认值对应"入门级数据中心 GPU"量级）。

    Attributes:
        prefill_ms_per_token: 预填充阶段每个 prompt token 的耗时（算力受限，线性叠加）。
        decode_base_ms: 每个解码步的固定开销，即读一遍模型权重的显存带宽成本。
            这是批处理收益的来源：它对整批**只付一次**。
        decode_per_request_ms: 每个解码步中与批大小线性相关的部分（attention 计算等）。
        max_batch_for_saturation: 权重摊薄的饱和批大小（超过后收益趋近饱和）。
        jitter: 单步耗时抖动比例（0 表示确定性，便于校验）。
    """

    prefill_ms_per_token: float = 0.22
    decode_base_ms: float = 9.0
    decode_per_request_ms: float = 0.55
    max_batch_for_saturation: int = 32
    jitter: float = 0.0

    def prefill_ms(self, total_prompt_tokens: int) -> float:
        return self.prefill_ms_per_token * max(total_prompt_tokens, 1)

    def weight_share(self, batch_size: int) -> float:
        """权重读取成本在批内的摊薄系数（B=1 时为 1.0，B 越大越接近 0）。"""
        b = max(batch_size, 1)
        saturation = max(self.max_batch_for_saturation, 1)
        return 1.0 / (1.0 + (b - 1) / saturation)

    def decode_step_ms(self, batch_size: int) -> float:
        """一个解码步的耗时（批内所有请求共享这一次开销）。"""
        b = max(batch_size, 1)
        return self.decode_base_ms * self.weight_share(b) + self.decode_per_request_ms * b

    def serial_step_ms(self, batch_size: int) -> float:
        """ "无批处理"等价耗时：每个请求各跑一次前向，成本线性相加。"""
        return self.decode_step_ms(1) * max(batch_size, 1)

    def theoretical_ratio(self, batch_size: int) -> float:
        """单步吞吐增益 = 串行成本 / 批处理成本（解析解，用于校验实测）。"""
        return self.serial_step_ms(batch_size) / self.decode_step_ms(batch_size)

    def theoretical_total_ms(
        self, batch_size: int, num_requests: int, prompt_tokens: int, max_tokens: int
    ) -> float:
        """给定批大小与请求数，理论总耗时（假设批被填满）。"""
        b = max(batch_size, 1)
        batches = num_requests / b
        per_batch = self.prefill_ms(prompt_tokens) * b + self.decode_step_ms(b) * max_tokens
        return batches * per_batch


def reference_hardware() -> SimulatedHardware:
    """参考硬件：RTX 4060 Laptop 8GB 量级（本项目实际可用的卡）。"""
    return SimulatedHardware(
        prefill_ms_per_token=0.22,
        decode_base_ms=9.0,
        decode_per_request_ms=0.55,
        max_batch_for_saturation=32,
        jitter=0.0,
    )


class _BatchRendezvous:
    """按解码步聚合的屏障：保证同批请求共享同一次解码开销。

    工作方式（每个解码步一次）：
    1. 每个请求调用 ``arrive()``，声明"我要生成第 i 个 token"；
    2. 记录本步的参与人数（= 批大小），并把未来交给它们等待；
    3. 当事件循环空闲时（``await asyncio.sleep(0)`` 之后），
       调度协程把本步的请求全部唤醒，并让**其中一个**承担真实的
       ``decode_step_ms(B)`` 等待时间，其余请求与它共享这次等待。

    这样"批越大，单位请求摊到的解码时间越短"就被如实建模了。
    """

    __slots__ = (
        "_last_batch_size",
        "_pending",
        "_step_waiter",
        "_task",
        "billing_ms",
        "step_wait_ms",
        "steps_completed",
    )

    def __init__(self) -> None:
        self._pending: list[asyncio.Future[None]] = []
        self._step_waiter: asyncio.Future[None] | None = None
        self._task: asyncio.Task[None] | None = None
        self._last_batch_size: int = 0
        # 真实累计值：每个解码步只累加一次（而不是每个等待者各算一次），
        # 否则 16 个请求会把同一次共享睡眠重复计 16 次，导致统计虚高。
        self.step_wait_ms = 0.0
        self.steps_completed = 0
        # 理论计费累计（纯公式，不受定时器精度影响）—— 用于测试与校验
        self.billing_ms = 0.0

    async def step(self, compute_step_ms: Callable[[int], float]) -> int:
        """登记并等待本解码步完成。

        Args:
            compute_step_ms: ``(batch_size) -> float``，根据本步真实参与人数返回耗时。

        Returns:
            本步的批大小（参与请求数）。
        """
        future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._pending.append(future)

        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run_step(compute_step_ms))

        await future
        return self._last_batch_size

    async def _run_step(self, compute_step_ms: Callable[[int], float]) -> None:
        # 让出事件循环，收集同一步内所有到达的请求
        await asyncio.sleep(0)
        waiters = self._pending
        self._pending = []
        batch_size = max(len(waiters), 1)
        self._last_batch_size = batch_size

        delay_ms = float(compute_step_ms(batch_size))
        started = asyncio.get_running_loop().time()
        if delay_ms > 0:
            await asyncio.sleep(delay_ms * _MS)
        # 只在此处累加一次真实耗时
        self.step_wait_ms += (asyncio.get_running_loop().time() - started) * 1000.0
        self.billing_ms += delay_ms
        self.steps_completed += 1

        for w in waiters:
            if not w.done():
                w.set_result(None)


class SimulatedBackend(InferenceBackend):
    """按耗时模型"消耗时间"的推理后端：不产生真实 token，但耗时量级与相对关系正确。

    与 ``EchoBackend`` 的分工：
    - ``EchoBackend``：每 token 延迟恒定且与并发无关 → 用来测**网关与协议层**的纯开销；
    - ``SimulatedBackend``：耗时随批大小非线性变化 → 用来测**调度引擎**的批处理收益。
    """

    name = "simulated"

    def __init__(self, hardware: SimulatedHardware | None = None) -> None:
        self.hw = hardware or reference_hardware()
        self.batches_seen: list[int] = []
        self.step_batch_sizes: list[int] = []  # 每个解码步的真实批大小（用于校验）
        # 每个"批窗口"（一次 _run_batch 派发的请求集合）的记录：
        # prefill 在批内是**并发**发生的，因此整批只付"单请求 prefill 时长"，
        # 而不是"N 个请求的 prefill 之和"。用批窗口记账才能得出正确的理论值。
        self.batch_records: list[dict[str, int]] = []
        self._active = 0
        self._peak_active = 0
        self._window = 0
        self._rendezvous = _BatchRendezvous()

    # ------------------------------------------------------------ 可观测量
    @property
    def peak_concurrency(self) -> int:
        """观测到的最大并发请求数（即引擎实际达成的批大小）。"""
        return self._peak_active

    @property
    def observed_peak_batch(self) -> int:
        return max(self.step_batch_sizes, default=0)

    @property
    def mean_step_batch(self) -> float:
        if not self.step_batch_sizes:
            return 0.0
        return sum(self.step_batch_sizes) / len(self.step_batch_sizes)

    @property
    def num_decode_steps(self) -> int:
        """真实发生的解码步数（每个融合步只算一次）。"""
        return self._rendezvous.steps_completed

    @property
    def decode_wait_ms(self) -> float:
        """解码阶段真实累计耗时（已消除重复计数）。"""
        return self._rendezvous.step_wait_ms

    @property
    def billing_ms(self) -> float:
        """解码阶段的**理论计费**累计（每步只算一次）。

        与 ``decode_wait_ms`` 的区别：
        - ``billing_ms`` 是耗时模型算出来的应然值（纯公式）；
        - ``decode_wait_ms`` 是真实 ``await`` 实测值（含 asyncio 定时器开销）。
        两者之比即"定时器开销比"，而它们的**累加方式**决定了串行/批处理的差异：
        串行后端按每请求一次前向计费，因此同一步的计费值大得多。
        """
        return self._rendezvous.billing_ms

    def batch_windows(self) -> dict[int, dict[str, int]]:
        """按批窗口聚合：``{window_id: {"batch_size": B, "prompt_tokens": P}}``。"""
        out: dict[int, dict[str, int]] = {}
        for rec in self.batch_records:
            out[rec["window"]] = {"batch_size": rec["batch_size"], "prompt_tokens": rec["prompt_tokens"]}
        return out

    def theoretical_prefill_ms(self) -> float:
        """解析 prefill 总耗时：每个批窗口只付一次"单请求 prefill 时长"。"""
        return sum(self.hw.prefill_ms(w["prompt_tokens"]) for w in self.batch_windows().values())

    def theoretical_decode_ms(self) -> float:
        """解析 decode 总耗时：每个解码步按当步批大小计费一次。"""
        return sum(self.hw.decode_step_ms(b) for b in self.step_batch_sizes)

    def estimated_gpu_utilization(self) -> float:
        """估算 GPU 利用率。

        定义：解码阶段中"真正用于请求计算"的时间占比。
        批大小为 1 时大部分时间花在读取权重（不可避免的空转）上，利用率低；
        批越大，同一份权重服务越多请求，利用率越高。
        这就是"批处理提升 GPU 利用率"的量化解释。
        """
        if not self.step_batch_sizes:
            return 0.0
        utils = []
        for b in self.step_batch_sizes:
            useful = self.hw.decode_per_request_ms * b
            utils.append(min(useful / max(self.hw.decode_step_ms(b), 1e-9), 1.0))
        return sum(utils) / len(utils)

    # ------------------------------------------------------------ 生成
    async def generate(
        self,
        request_id: str,
        messages: list[ChatMessage],
        params: GenerateParams,
    ) -> AsyncIterator[TokenChunk]:
        prompt_tokens = self.estimate_prompt_tokens(messages)
        self._active += 1
        self._peak_active = max(self._peak_active, self._active)
        if self._active == 1:
            # 批窗口开始：0 → 1 的跃迁标识一个新批次（同一批的并发请求共享一个窗口号）
            self._window += 1
        self.batch_records.append(
            {"window": self._window, "batch_size": self._active, "prompt_tokens": prompt_tokens}
        )
        try:
            # ---- Prefill：批内并发，因此整批只付一次"单请求 prefill 时长"
            prefill_ms = self.hw.prefill_ms(prompt_tokens)
            if prefill_ms > 0:
                await asyncio.sleep(prefill_ms * _MS)

            # ---- Decode：逐 token，每步与同批请求**同步**共享一次解码开销
            produced = 0
            for _ in range(params.max_tokens):
                batch_size = await self._rendezvous.step(self.hw.decode_step_ms)
                self.step_batch_sizes.append(batch_size)
                if batch_size not in self.batches_seen:
                    self.batches_seen.append(batch_size)
                produced += 1
                yield TokenChunk(text="x", prompt_tokens=prompt_tokens, completion_tokens=produced)

            yield TokenChunk(
                text="",
                finish_reason="length",
                prompt_tokens=prompt_tokens,
                completion_tokens=produced,
            )
        finally:
            self._active -= 1

    async def health(self) -> dict[str, object]:
        return {
            "backend": self.name,
            "ready": True,
            "hardware": {
                "prefill_ms_per_token": self.hw.prefill_ms_per_token,
                "decode_base_ms": self.hw.decode_base_ms,
                "decode_per_request_ms": self.hw.decode_per_request_ms,
            },
            "observed_peak_batch": self.observed_peak_batch,
            "mean_step_batch": round(self.mean_step_batch, 3),
            "estimated_gpu_utilization": round(self.estimated_gpu_utilization(), 4),
        }


class SerialReferenceBackend(SimulatedBackend):
    """串行对照后端：解码步永远按"单请求前向"计费，模拟"没有批处理"的朴素实现。

    实现方式：批同步仍然发生（保证同一步的请求一起推进），
    但耗时按 ``decode_step_ms(1) × 参与人数`` 计算 ——
    等价于"每个请求各跑一次独立前向"，因此它是评估批处理收益的正确基线。
    """

    name = "simulated-serial"

    async def generate(
        self,
        request_id: str,
        messages: list[ChatMessage],
        params: GenerateParams,
    ) -> AsyncIterator[TokenChunk]:
        prompt_tokens = self.estimate_prompt_tokens(messages)
        self._active += 1
        self._peak_active = max(self._peak_active, self._active)
        if self._active == 1:
            self._window += 1
        self.batch_records.append(
            {"window": self._window, "batch_size": self._active, "prompt_tokens": prompt_tokens}
        )
        try:
            prefill_ms = self.hw.prefill_ms(prompt_tokens)
            if prefill_ms > 0:
                await asyncio.sleep(prefill_ms * _MS)

            produced = 0
            for _ in range(params.max_tokens):
                # 关键差异：按"每请求一次前向"计费
                batch_size = await self._rendezvous.step(self.hw.serial_step_ms)
                self.step_batch_sizes.append(batch_size)
                produced += 1
                yield TokenChunk(text="x", prompt_tokens=prompt_tokens, completion_tokens=produced)

            yield TokenChunk(
                text="",
                finish_reason="length",
                prompt_tokens=prompt_tokens,
                completion_tokens=produced,
            )
        finally:
            self._active -= 1


__all__ = [
    "SerialReferenceBackend",
    "SimulatedBackend",
    "SimulatedHardware",
    "reference_hardware",
]
