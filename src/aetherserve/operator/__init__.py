"""Kubernetes Operator 包：CRD 模型、清单构建、调和逻辑与 kopf 处理器。

模块分层（依赖单向）
-------------------
    crd.py        领域模型 + CRD schema + 校验     （无依赖）
    builders.py   领域模型 → K8s 清单               （依赖 crd）
    canary.py     灰度状态机                        （依赖 crd）
    reconciler.py 求差与决策（纯逻辑，可完整单测）    （依赖 builders/canary/crd）
    handlers.py   kopf 处理器 + 集群读写             （依赖 reconciler）
    main.py       进程入口                          （依赖 handlers）
"""

from .crd import (
    API_GROUP,
    API_VERSION,
    KIND,
    PLURAL,
    InferenceServiceSpec,
    InferenceServiceStatus,
    parse_spec,
    validate_spec,
)

__all__ = [
    "API_GROUP",
    "API_VERSION",
    "KIND",
    "PLURAL",
    "InferenceServiceSpec",
    "InferenceServiceStatus",
    "parse_spec",
    "validate_spec",
]
