"""校验文档中引用的数字与压测原始数据是否一致。

为什么需要这个校验？
-------------------
文档与报告里的每个关键数字都必须能被原始数据复现。
若文档之间出现不一致（例如报告写 4.9× 而数据算出来是 3.2×），
评审或排查时被追问就会立刻暴露。因此把"数字对账"自动化。

校验对象：
- `README.md` 的性能数据表
- `docs/性能测试报告.md` 中的关键结论

数据来源约定（重要）
-------------------
`bench/results/` 里存放的是**已发布的基准数据**，文档中的数字必须与它一致。
因此本脚本不仅比对数字，还**校验数据文件的实验参数**是否是文档声明的那一套
（并发梯度、请求数、prompt/生成长度）—— 否则"用不同参数跑出来的数据"
会让对账结果毫无意义（这是实际踩过的坑：CI 的快速 smoke 测试曾把
缩减参数的结果写进同一目录，导致对账误报）。
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"

# 文档中性能数据表的实验参数（必须与数据文件一致）
EXPECTED_PARAMS = {
    "concurrency": [1, 4, 8, 16, 32],
    "requests": 64,
    "max_tokens": 32,
    "prompt_tokens": 256,
}


def load_latest(pattern: str) -> dict:
    files = sorted(glob.glob(str(RESULTS / pattern)))
    if not files:
        print(f"!! 未找到匹配 {pattern} 的数据文件")
        sys.exit(2)
    return json.loads(Path(files[-1]).read_text(encoding="utf-8"))


def check(label: str, claimed: float, actual: float, tol: float, unit: str = "") -> bool:
    ok = abs(actual - claimed) <= tol
    mark = "OK  " if ok else "FAIL"
    print(f"  [{mark}] {label}: 文档称 {claimed}{unit} / 实测 {actual:.4g}{unit} (容差 {tol})")
    return ok


def check_experiment_params(payload: dict, label: str) -> bool:
    """校验数据文件的实验参数是否与文档声明一致。

    这是一个"元校验"：如果参数对不上，说明取到了错误的数据文件，
    后面的数字比对无论通过与否都没有意义。
    """
    args = payload.get("args", {})
    ok = True
    print(f"  [{label}] 实验参数核对（文档声明的配置）")
    for key, expected in EXPECTED_PARAMS.items():
        actual = args.get(key)
        match = actual == expected
        if not match:
            ok = False
        mark = "OK  " if match else "FAIL"
        print(f"    [{mark}] {key}: 期望 {expected} / 数据文件 {actual}")
    return ok


def main() -> int:
    all_ok = True

    sim = load_latest("*-sim.json")
    rows = {r["label"]: r for r in sim["results"]}
    serial16 = rows["serial(no-batching) c=16"]
    batch16 = rows["batched b=16 c=16"]
    ratio = batch16["qps"] / serial16["qps"]

    echo = load_latest("*-echo.json")
    e64 = next(r for r in echo["results"] if r["concurrency"] == 64)

    ver = json.loads((RESULTS / "model_verification.json").read_text(encoding="utf-8"))

    print("=" * 78)
    print("文档数字对账（README / 性能报告 vs 压测原始数据）")
    print("=" * 78)
    print(f"数据来源: {sim['environment']['timestamp']} / {echo['environment']['timestamp']}")
    print()

    print("[第 0 步] 元校验：数据文件的实验参数是否与文档声明一致")
    all_ok &= check_experiment_params(sim, "sim")
    print()

    print("[调度引擎]")
    all_ok &= check("串行 QPS", 4.63, serial16["qps"], 0.05)
    all_ok &= check("批处理 QPS", 22.60, batch16["qps"], 0.05)
    all_ok &= check("吞吐提升倍数", 4.9, ratio, 0.1, "×")
    all_ok &= check("串行 GPU 利用率", 5.8, serial16["gpu_utilization"] * 100, 0.5, "%")
    all_ok &= check("批处理 GPU 利用率", 58.9, batch16["gpu_utilization"] * 100, 0.5, "%")

    print("\n[流式网关]")
    all_ok &= check("网关 QPS @64 并发", 1603, e64["qps"], 2.0)
    all_ok &= check("TTFT P99 @64 并发", 10.5, e64["ttft"]["p99_ms"], 0.5, "ms")

    print("\n[工程质量]")
    print(f"  [INFO] 自校验偏差（报告写 1.56%，本次运行为 {ver['deviation_pct']}%，")
    print("         报告已声明该值每次运行有波动，阈值 15%）")
    assert ver["verdict"] == "PASS", "自校验未通过！"
    print(f"  [OK  ] 自校验判定: {ver['verdict']}")

    print()
    print("=" * 78)
    print("结论:", "全部数字可复现 ✅" if all_ok else "存在不一致，需修正文档 ❌")
    print("=" * 78)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
