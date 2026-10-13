"""压测入口：产出可复现的性能数据。

用法::

    # A) 无 GPU 仿真模式：量化连续批处理的调度收益（默认）
    python bench/run_bench.py --mode sim

    # B) Echo 后端：只测网关/协议层纯开销（不含批处理效应）
    python bench/run_bench.py --mode echo

    # C) 真实 GPU：需要先启动 vLLM（见 README）
    python bench/run_bench.py --mode vllm --endpoint http://127.0.0.1:8001

输出:
    bench/results/<timestamp>-<mode>.json    原始数据
    bench/results/<timestamp>-<mode>.md      Markdown 表格（可直接嵌入报告）
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

# 允许直接以脚本方式运行（无需先安装）
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aetherserve.backends.echo_backend import EchoBackend
from aetherserve.config import Settings
from aetherserve.sim import (
    LoadTestResult,
    SerialReferenceBackend,
    SimulatedBackend,
    reference_hardware,
    run_load_test,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def base_settings(**overrides: Any) -> Settings:
    s = Settings()
    s.environment = "bench"
    s.batching.max_batch_size = int(overrides.pop("max_batch_size", 16))
    s.batching.max_batch_tokens = int(overrides.pop("max_batch_tokens", 16384))
    s.batching.max_wait_ms = float(overrides.pop("max_wait_ms", 20.0))
    s.batching.worker_concurrency = int(overrides.pop("worker_concurrency", 4))
    s.batching.max_queue_size = int(overrides.pop("max_queue_size", 4096))
    s.batching.max_queue_tokens = int(overrides.pop("max_queue_tokens", 1_000_000))
    s.batching.aging_ms = 500.0
    s.backend.model = "bench-model"
    s.backend.backend = "simulated"
    s.backend.default_max_tokens = 64
    for k, v in overrides.items():
        setattr(s.batching, k, v)
    return s


def environment_info() -> dict[str, Any]:
    """记录测试环境，保证数据可复现（性能数据脱离环境就没有意义）。"""
    import dataclasses
    import os

    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware_model": {
            "note": "耗时模型（非真实 GPU 实测），量级对应入门级数据中心 GPU",
            **dataclasses.asdict(reference_hardware()),
        },
    }


# --------------------------------------------------------------------- 场景
async def scenario_sim(args: argparse.Namespace) -> list[LoadTestResult]:
    """仿真对照：串行（无批处理） vs 连续批处理。

    **对比口径（很关键，否则数字会骗人）**
    ------------------------------------
    朴素做法是"串行组给高并发、批处理组给低并发"，这样比出来的 QPS 差异里
    混进了"并行度不同"的因素，结论不可信。这里采用两个口径同时报告：

    1. **同并发窗口对比**（主口径）：每个并发档位都跑一组串行基线，
       两组使用相同的在途请求数。此时唯一的变量是"一个解码步服务几个请求"，
       因此 ``归一化吞吐增益 = (批处理QPS / 串行QPS) / (批处理并发 / 串行并发)``
       恰好反映单步效率提升，可直接与耗时模型的 ``decode_step_ms`` 比值对照验证。
    2. 绝对 QPS：反映真实部署中"同一张卡能扛多少业务量"。

    两个口径都保留在报告里，避免用单一数字掩盖真实情况。
    """
    results: list[LoadTestResult] = []
    hw = reference_hardware()

    for concurrency in args.concurrency:
        # ---- 对照组：无批处理。并发窗口限制为 1，保证"一次只处理一个请求"
        serial = SerialReferenceBackend(hardware=hw)
        results.append(
            await run_load_test(
                serial,
                base_settings(max_batch_size=1, max_wait_ms=0.0, worker_concurrency=1),
                label=f"serial(no-batching) c={concurrency}",
                concurrency=concurrency,
                requests=args.requests,
                max_tokens=args.max_tokens,
                prompt_tokens=args.prompt_tokens,
            )
        )

        # ---- 实验组：连续批处理。批大小上限取 min(配置值, 并发)，避免批不满时的假象。
        #
        # 注意 worker_concurrency **必须为 1**：真实引擎的连续批处理是
        # "把所有活跃请求融合进同一个前向计算"，而不是"同时跑多个批"。
        # 若允许并发批，仿真后端会把"多个批的并发总和"误当成批大小，
        # 得到的耗时曲线不具物理意义（批大小扫描会出现非单调的假峰）。
        batched = SimulatedBackend(hardware=hw)
        batch_size = max(1, min(args.batch_size, concurrency))
        results.append(
            await run_load_test(
                batched,
                base_settings(max_batch_size=batch_size, worker_concurrency=1),
                label=f"batched b={batch_size} c={concurrency}",
                concurrency=concurrency,
                requests=args.requests,
                max_tokens=args.max_tokens,
                prompt_tokens=args.prompt_tokens,
            )
        )

    return results


async def scenario_batch_sweep(args: argparse.Namespace) -> list[LoadTestResult]:
    """批大小扫描：观察吞吐随批大小的变化曲线与饱和点。

    worker_concurrency 固定为 1 —— 理由同 ``scenario_sim``：
    连续批处理是"一个融合前向"，并发批会让仿真把总并发误当批大小。
    """
    results: list[LoadTestResult] = []
    hw = reference_hardware()
    for batch_size in args.batch_sweep:
        backend = SimulatedBackend(hardware=hw)
        results.append(
            await run_load_test(
                backend,
                base_settings(max_batch_size=batch_size, worker_concurrency=1),
                label=f"batch_size={batch_size}",
                concurrency=args.sweep_concurrency,
                requests=args.requests,
                max_tokens=args.max_tokens,
                prompt_tokens=args.prompt_tokens,
            )
        )
    return results


async def scenario_echo(args: argparse.Namespace) -> list[LoadTestResult]:
    """Echo 后端：测网关与协议层的纯开销（不含批处理效应），作为延迟下界。"""
    results: list[LoadTestResult] = []
    for concurrency in args.concurrency:
        backend = EchoBackend(token_delay_ms=0.5, max_tokens_hint=args.max_tokens)
        results.append(
            await run_load_test(
                backend,
                base_settings(max_batch_size=args.batch_size),
                label=f"echo(gateway-only) c={concurrency}",
                concurrency=concurrency,
                requests=args.requests,
                max_tokens=args.max_tokens,
                prompt_tokens=args.prompt_tokens,
            )
        )
    return results


async def scenario_vllm(args: argparse.Namespace) -> list[LoadTestResult]:
    """真实 GPU：通过 vLLM 的 OpenAI 兼容接口驱动（需要先启动 vLLM 服务）。"""
    from aetherserve.backends.vllm_backend import VLLMBackend

    results: list[LoadTestResult] = []
    for concurrency in args.concurrency:
        backend = VLLMBackend(
            endpoint=args.endpoint,
            model=args.model,
            timeout_s=600.0,
        )
        settings = base_settings(max_batch_size=args.batch_size)
        settings.backend.backend = "vllm"
        settings.backend.model = args.model
        try:
            results.append(
                await run_load_test(
                    backend,
                    settings,
                    label=f"vllm c={concurrency}",
                    concurrency=concurrency,
                    requests=args.requests,
                    max_tokens=args.max_tokens,
                    prompt_tokens=args.prompt_tokens,
                )
            )
        finally:
            await backend.aclose()
    return results


# --------------------------------------------------------------------- 报告
def markdown_report(mode: str, env: dict[str, Any], results: list[LoadTestResult]) -> str:
    lines: list[str] = []
    lines.append(f"# 性能测试报告（{mode} 模式）\n")
    lines.append("## 测试环境\n")
    lines.append(f"- 运行时间：{env['timestamp']}")
    lines.append(f"- 平台：{env['platform']}")
    lines.append(f"- Python：{env['python']}，CPU 逻辑核：{env['cpu_count']}")
    if "hardware_model" in env:
        hw = env["hardware_model"]
        lines.append(
            f"- 耗时模型：prefill={hw['prefill_ms_per_token']}ms/token，"
            f"decode_base={hw['decode_base_ms']}ms/step，"
            f"decode_per_request={hw['decode_per_request_ms']}ms/step"
        )
        if "note" in hw:
            lines.append(f"- 说明：{hw['note']}")
    lines.append("")

    lines.append("## 汇总结果\n")
    lines.append(
        "| 场景 | 并发 | 请求数 | QPS | tokens/s | 峰值批大小 | 平均批大小 | TTFT P50 | TTFT P99 | TPOT P50 | E2E P99 | 成功率 |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        lines.append(
            f"| {r.label} | {r.concurrency} | {r.completed}/{r.requests} | {r.qps} | {r.tokens_per_second} | "
            f"{r.observed_peak_batch} | {r.mean_batch_size:.2f} | {r.ttft.p50_ms:.1f}ms | {r.ttft.p99_ms:.1f}ms | "
            f"{r.tpot.p50_ms:.1f}ms | {r.e2e.p99_ms:.1f}ms | {r.success_rate:.2%} |"
        )
    lines.append("")

    # 自动计算批处理收益（对照组与实验组成对出现时）
    gains: list[str] = []
    by_conc: dict[int, dict[str, LoadTestResult]] = {}
    for r in results:
        kind = (
            "serial"
            if r.label.startswith("serial")
            else ("batched" if r.label.startswith("batched") else None)
        )
        if kind:
            by_conc.setdefault(r.concurrency, {})[kind] = r

    hw_info = env.get("hardware_model", {})
    model_base = float(hw_info.get("decode_base_ms", 0) or 0)
    model_per_req = float(hw_info.get("decode_per_request_ms", 0) or 0)
    model_sat = float(hw_info.get("max_batch_for_saturation", 1) or 1)

    def model_step_ms(b: int) -> float:
        share = 1.0 / (1.0 + (max(b, 1) - 1) / max(model_sat, 1))
        return model_base * share + model_per_req * max(b, 1)

    for conc, pair in sorted(by_conc.items()):
        if "serial" not in pair or "batched" not in pair:
            continue
        s, bt = pair["serial"], pair["batched"]
        qps_gain = bt.qps / s.qps if s.qps else 0.0
        # 归一化：剔除"并行度不同"带来的影响，得到单步效率增益
        concurrency_ratio = (bt.concurrency / s.concurrency) if s.concurrency else 1.0
        normalized = qps_gain / concurrency_ratio if concurrency_ratio else 0.0

        step_serial = model_step_ms(s.concurrency)
        step_batched = model_step_ms(bt.concurrency)
        expected = (step_serial / step_batched) if step_batched else 0.0
        deviation = (abs(normalized - expected) / expected * 100) if expected else 0.0

        line = (
            f"- 并发窗口 {conc}：绝对 QPS {s.qps} → **{bt.qps}**"
            f"（{qps_gain:.2f}×，其中 {concurrency_ratio:.0f}× 来自并行度）\n"
            f"  - **归一化吞吐增益 {normalized:.2f}×**（剔除并行度差异）"
        )
        if expected:
            line += f"，理论期望 {expected:.2f}×，偏差 {deviation:.1f}%"
        line += (
            f"\n  - 延迟：TTFT P99 {s.ttft.p99_ms:.0f}ms → {bt.ttft.p99_ms:.0f}ms；"
            f"GPU 利用率 {s.gpu_utilization:.1%} → {bt.gpu_utilization:.1%}"
        )
        gains.append(line)

    if gains:
        lines.append("## 连续批处理收益\n")
        lines.append(
            "> 归一化吞吐增益 = (批处理 QPS ÷ 串行 QPS) ÷ (批处理并发 ÷ 串行并发)，"
            "它剔除了并行度差异，反映**单个解码步的效率提升**，可与耗时模型理论值互相校验。\n"
        )
        lines.extend(gains)
        lines.append("")

    lines.append("## 原始数据\n")
    lines.append("```json")
    lines.append(json.dumps([r.to_dict() for r in results], ensure_ascii=False, indent=2))
    lines.append("```")
    return "\n".join(lines)


# --------------------------------------------------------------------- 主流程
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AetherServe 压测工具")
    p.add_argument("--mode", choices=["sim", "sweep", "echo", "vllm"], default="sim")
    p.add_argument("--concurrency", type=lambda s: [int(x) for x in s.split(",")], default=[1, 4, 16, 32])
    p.add_argument(
        "--batch-sweep", type=lambda s: [int(x) for x in s.split(",")], default=[1, 2, 4, 8, 16, 32]
    )
    p.add_argument("--sweep-concurrency", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--worker-concurrency", type=int, default=4)
    p.add_argument("--requests", type=int, default=48, help="每个场景的请求总数")
    p.add_argument("--max-tokens", type=int, default=32, help="每个请求生成的最大 token 数")
    p.add_argument("--prompt-tokens", type=int, default=128, help="每个请求的 prompt token 数")
    p.add_argument("--endpoint", default="http://127.0.0.1:8001", help="vLLM 服务地址")
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct-AWQ")
    p.add_argument("--out-dir", default=str(RESULTS_DIR))
    return p.parse_args(argv)


async def async_main(args: argparse.Namespace) -> list[LoadTestResult]:
    if args.mode == "sim":
        return await scenario_sim(args)
    if args.mode == "sweep":
        return await scenario_batch_sweep(args)
    if args.mode == "echo":
        return await scenario_echo(args)
    return await scenario_vllm(args)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    env = environment_info()

    print(
        f"[bench] 模式={args.mode} 并发={args.concurrency} 请求数={args.requests} "
        f"max_tokens={args.max_tokens} prompt_tokens={args.prompt_tokens}"
    )

    results = asyncio.run(async_main(args))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    json_path = out_dir / f"{stamp}-{args.mode}.json"
    md_path = out_dir / f"{stamp}-{args.mode}.md"

    payload = {
        "mode": args.mode,
        "environment": env,
        "args": vars(args),
        "results": [r.to_dict() for r in results],
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(markdown_report(args.mode, env, results), encoding="utf-8")

    # 控制台摘要
    print("\n" + "=" * 118)
    print(
        f"{'场景':<32}{'并发':>5}{'QPS':>9}{'tok/s':>10}{'峰值批':>7}{'TTFT P50':>10}{'TTFT P99':>10}{'E2E P99':>10}{'成功率':>8}"
    )
    print("-" * 118)
    for r in results:
        print(
            f"{r.label:<32}{r.concurrency:>5}{r.qps:>9.2f}{r.tokens_per_second:>10.1f}"
            f"{r.observed_peak_batch:>7}{r.ttft.p50_ms:>9.1f}ms{r.ttft.p99_ms:>9.1f}ms"
            f"{r.e2e.p99_ms:>9.1f}ms{r.success_rate:>8.1%}"
        )
    print("=" * 118)
    print(f"\n[bench] 原始数据: {json_path}")
    print(f"[bench] 报告:     {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
