"""校验简历中引用的数字与压测原始数据是否一致。

为什么需要这个校验？
-------------------
简历与性能报告中的每个数字都必须能被原始数据复现。
若文档间出现不一致（例如简历写 4.9× 而数据算出来是 3.2×），
面试时被追问就会立刻暴露。因此把"数字对账"自动化。
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"


def load_latest(pattern: str) -> dict:
    files = sorted(glob.glob(str(RESULTS / pattern)))
    if not files:
        print(f"!! 未找到匹配 {pattern} 的数据文件")
        sys.exit(2)
    return json.loads(Path(files[-1]).read_text(encoding="utf-8"))


def check(label: str, claimed: float, actual: float, tol: float, unit: str = "") -> bool:
    ok = abs(actual - claimed) <= tol
    mark = "OK  " if ok else "FAIL"
    print(f"  [{mark}] {label}: 简历称 {claimed}{unit} / 实测 {actual:.4g}{unit} (容差 {tol})")
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
    print("简历数字对账（简历 vs 压测原始数据）")
    print("=" * 78)
    print(f"数据来源: {sim['environment']['timestamp']} / {echo['environment']['timestamp']}")
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
