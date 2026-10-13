#!/usr/bin/env python
"""生成部署清单：CRD 与示例 InferenceService。

用法::

    python deploy/k8s/generate_manifests.py            # 打印 CRD 到 stdout
    python deploy/k8s/generate_manifests.py --example  # 打印示例 CR
    python deploy/k8s/generate_manifests.py --all      # 打印全部（CRD + 示例）

为什么要用脚本生成而不是手写 YAML？
---------------------------------
CRD 的 schema 手写极易与代码脱节（改了 dataclass 忘了改 YAML）。
本脚本从 ``aetherserve.operator.crd.crd_manifest()`` 直接推导，
保证"代码即契约"，也便于 CI 校验（`make crd` 前后 diff 为空则说明一致）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import yaml

from aetherserve.operator.crd import API_GROUP, API_VERSION, KIND, crd_manifest


def example_inferenceservice() -> dict[str, Any]:
    """示例 CR：演示最常用的字段组合（含灰度与自定义指标扩缩容）。"""
    return {
        "apiVersion": f"{API_GROUP}/{API_VERSION}",
        "kind": KIND,
        "metadata": {
            "name": "qwen-1-5b",
            "namespace": "ai-serving",
            "labels": {"app.kubernetes.io/name": "aetherserve"},
        },
        "spec": {
            "model": "Qwen/Qwen2.5-1.5B-Instruct-AWQ",
            "backend": "vllm",
            "quantization": "awq",
            "image": "ghcr.io/aetherserve/gateway:0.1.0",
            "replicas": 2,
            "port": 8000,
            "maxModelLen": 4096,
            "gpuMemoryUtilization": 0.85,
            # 批处理参数：按模型规模与 SLA 调整（参见 docs/性能测试报告.md 的调参建议）
            "maxBatchSize": 16,
            "maxBatchTokens": 8192,
            "maxWaitMs": 20,
            "resources": {
                "cpuRequest": "1000m",
                "memoryRequest": "2Gi",
                "cpuLimit": "4000m",
                "memoryLimit": "8Gi",
                "gpu": 1,
            },
            "autoscaling": {
                "enabled": True,
                "minReplicas": 2,
                "maxReplicas": 8,
                # 队列深度反应最快（提前扩容），KV Cache 水位最能反映真实瓶颈（精准收敛）
                "metrics": ["queue_depth", "kv_cache"],
                "targetQueueDepth": 16,
                "targetKvCacheUtilization": 0.8,
                "scaleDownStabilizationS": 300,
            },
            "canary": {
                "enabled": True,
                "initialWeight": 10,
                "step": 20,
                "maxErrorRate": 0.05,
                "minSamples": 100,
            },
            "serviceMonitor": True,
            "nodeSelector": {"aetherserve.io/accelerator": "nvidia"},
        },
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="生成 AetherServe 部署清单")
    p.add_argument("--example", action="store_true", help="只输出示例 InferenceService")
    p.add_argument("--all", action="store_true", help="输出 CRD 与示例（用 --- 分隔）")
    p.add_argument("--out", help="写入文件而不是打印到 stdout")
    args = p.parse_args(argv)

    docs: list[dict[str, Any]] = []
    if not args.example:
        docs.append(crd_manifest())
    if args.example or args.all:
        docs.append(example_inferenceservice())

    rendered = yaml.safe_dump_all(docs, allow_unicode=True, sort_keys=False, default_flow_style=False)

    if args.out:
        Path(args.out).write_text(rendered, encoding="utf-8")
        print(f"已写入 {args.out}")
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
