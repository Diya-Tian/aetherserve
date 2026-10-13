"""``InferenceService`` 自定义资源（CRD）的领域模型。

为什么用 dataclass 而不是直接操作 dict？
---------------------------------------
清单生成是控制面的**核心逻辑**，需要：
1. **编译期可检查**：字段拼写错误在测试阶段就暴露，而不是等到 apply 到集群才报错；
2. **可测试**：纯数据类可以脱离集群做单元测试（本项目的 Operator 测试全部不依赖真实集群）；
3. **可生成 CRD schema**：由模型推导出 OpenAPI schema，避免 `crd.yaml` 与代码脱节。

命名规范遵循 Kubernetes API 惯例：
    apiVersion: aetherserve.io/v1alpha1
    kind: InferenceService
    spec.replicas / spec.autoscaling / spec.canary ...
    status.conditions[] 使用标准 Condition 三元组（type/status/reason）
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

API_GROUP = "aetherserve.io"
API_VERSION = "v1alpha1"
KIND = "InferenceService"
PLURAL = "inferenceservices"

DEFAULT_IMAGE = "ghcr.io/aetherserve/gateway:0.1.0"
DEFAULT_PORT = 8000
DEFAULT_METRICS_PORT = 8000

ConditionType = Literal["Ready", "Progressing", "Degraded", "CanaryRunning", "CanaryFailed"]
ConditionStatus = Literal["True", "False", "Unknown"]


# --------------------------------------------------------------------- 工具
def slugify(value: str, max_len: int = 40) -> str:
    """把任意字符串转成合法的 Kubernetes 名称片段（RFC 1123 DNS label）。

    规则：小写、非字母数字替换为 ``-``、去掉首尾 ``-``、长度截断。
    模型名里的 ``/`` 与 ``.`` 都会在这里被规范化，避免生成非法资源名。
    """
    s = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    if not s:
        s = "x"
    return s[:max_len].rstrip("-")


def canonical_name(name: str, role: str | None = None) -> str:
    """生成资源名：``<cr-name>[-role]``。"""
    base = slugify(name, 40)
    return f"{base}-{role}" if role else base


def spec_hash(payload: dict[str, Any], length: int = 10) -> str:
    """对 spec 做稳定哈希，用于滚动更新与变更检测。

    为什么需要它？
    - **触发滚动更新**：把哈希写入 Pod 模板 annotation，
      这样只有 spec 真正变化（而非调和次数）才会触发 Deployment 滚动；
    - **幂等判定**：调和循环用 ``hash(期望) == hash(实际)`` 快速判断"是否需要动"。
    """
    import json

    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:length]


# --------------------------------------------------------------------- 子结构
@dataclass(slots=True)
class ResourceRequirements:
    """容器资源请求与上限。GPU 走扩展资源 ``nvidia.com/gpu``。"""

    cpu_request: str = "500m"
    memory_request: str = "512Mi"
    cpu_limit: str = "2000m"
    memory_limit: str = "2Gi"
    gpu: int = 0

    def requests(self) -> dict[str, str]:
        out = {"cpu": self.cpu_request, "memory": self.memory_request}
        if self.gpu > 0:
            out["nvidia.com/gpu"] = str(self.gpu)
        return out

    def limits(self) -> dict[str, str]:
        out = {"cpu": self.cpu_limit, "memory": self.memory_limit}
        if self.gpu > 0:
            out["nvidia.com/gpu"] = str(self.gpu)
        return out


@dataclass(slots=True)
class AutoscalingSpec:
    """弹性伸缩配置。

    ``metrics`` 支持两类来源：
    - ``queue_depth``：网关队列深度（爬坡前兆，反应最快）；
    - ``kv_cache``：KV Cache 显存水位（真实瓶颈，反应慢但准确）。
    同时使用两者可实现"提前扩容 + 精准收敛"。
    """

    enabled: bool = True
    min_replicas: int = 1
    max_replicas: int = 5
    target_queue_depth: int = 16
    target_kv_cache_utilization: float = 0.8
    scale_down_stabilization_s: int = 300
    metrics: list[str] = field(default_factory=lambda: ["queue_depth"])


@dataclass(slots=True)
class CanarySpec:
    """灰度发布配置：权重分批推进，失败自动回滚。"""

    enabled: bool = False
    initial_weight: int = 10
    step: int = 20
    max_error_rate: float = 0.05
    min_samples: int = 100
    analysis_interval_s: int = 60


@dataclass(slots=True)
class InferenceServiceSpec:
    """``InferenceService.spec``。"""

    model: str
    image: str = DEFAULT_IMAGE
    replicas: int = 1
    port: int = DEFAULT_PORT
    backend: str = "vllm"
    quantization: str | None = None
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.85

    # 批处理参数（透传给网关的环境变量，实现"按模型调参"）
    max_batch_size: int = 16
    max_batch_tokens: int = 8192
    max_wait_ms: float = 20.0

    resources: ResourceRequirements = field(default_factory=ResourceRequirements)
    autoscaling: AutoscalingSpec = field(default_factory=AutoscalingSpec)
    canary: CanarySpec = field(default_factory=CanarySpec)

    env: dict[str, str] = field(default_factory=dict)
    node_selector: dict[str, str] = field(default_factory=dict)
    service_monitor: bool = True

    def gateway_env(self) -> dict[str, str]:
        """把 spec 翻译成网关容器环境变量（``AETHERSERVE_`` 前缀）。"""
        env = {
            "AETHERSERVE_BACKEND": self.backend,
            "AETHERSERVE_MODEL": self.model,
            "AETHERSERVE_MAX_MODEL_LEN": str(self.max_model_len),
            "AETHERSERVE_GPU_MEMORY_UTILIZATION": str(self.gpu_memory_utilization),
            "AETHERSERVE_BATCH_MAX_BATCH_SIZE": str(self.max_batch_size),
            "AETHERSERVE_BATCH_MAX_BATCH_TOKENS": str(self.max_batch_tokens),
            "AETHERSERVE_BATCH_MAX_WAIT_MS": str(self.max_wait_ms),
        }
        if self.quantization:
            env["AETHERSERVE_QUANTIZATION"] = self.quantization
        env.update(self.env)
        return env

    def hash(self) -> str:
        """spec 的稳定指纹，用于触发滚动更新与幂等判定。"""
        return spec_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Condition:
    """标准 Kubernetes 条件三元组 + 可读原因。"""

    type: ConditionType
    status: ConditionStatus
    reason: str = ""
    message: str = ""
    last_transition_time: str = ""

    def to_dict(self) -> dict[str, str]:
        # 显式标注为 dict[str, str]：type/status 是 Literal[str]，而 dict 的值类型不变，
        # 不加注解会让 mypy 推断出 Literal 联合类型并在 return 处报不兼容。
        out: dict[str, str] = {"type": self.type, "status": self.status}
        if self.reason:
            out["reason"] = self.reason
        if self.message:
            out["message"] = self.message
        if self.last_transition_time:
            out["lastTransitionTime"] = self.last_transition_time
        return out


@dataclass(slots=True)
class InferenceServiceStatus:
    """``InferenceService.status``：观测到的实际状态。"""

    observed_generation: int = 0
    replicas: int = 0
    ready_replicas: int = 0
    canary_weight: int = 0
    canary_state: str = "None"  # None | Progressing | Promoted | RolledBack
    conditions: list[Condition] = field(default_factory=list)
    endpoint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "observedGeneration": self.observed_generation,
            "replicas": self.replicas,
            "readyReplicas": self.ready_replicas,
            "canaryWeight": self.canary_weight,
            "canaryState": self.canary_state,
            "endpoint": self.endpoint,
            "conditions": [c.to_dict() for c in self.conditions],
        }

    def set_condition(self, cond: Condition) -> None:
        """插入或替换同类型条件（保证 conditions 中每个 type 唯一）。"""
        self.conditions = [c for c in self.conditions if c.type != cond.type]
        self.conditions.append(cond)

    def get_condition(self, cond_type: str) -> Condition | None:
        for c in self.conditions:
            if c.type == cond_type:
                return c
        return None

    @property
    def is_ready(self) -> bool:
        cond = self.get_condition("Ready")
        return cond is not None and cond.status == "True"


# --------------------------------------------------------------------- CRD schema
def crd_manifest() -> dict[str, Any]:
    """生成 ``InferenceService`` 的 CRD 清单（结构性 schema）。

    手写 schema 的好处：它同时也是**接口文档** ——
    `kubectl explain inferenceservice.spec.canary` 能直接读出每个字段含义。
    """
    return {
        "apiVersion": "apiextensions.k8s.io/v1",
        "kind": "CustomResourceDefinition",
        "metadata": {"name": f"{PLURAL}.{API_GROUP}"},
        "spec": {
            "group": API_GROUP,
            "scope": "Namespaced",
            "names": {
                "plural": PLURAL,
                "singular": "inferenceservice",
                "kind": KIND,
                "shortNames": ["isvc"],
                "categories": ["ai", "serving"],
            },
            "versions": [
                {
                    "name": API_VERSION,
                    "served": True,
                    "storage": True,
                    "additionalPrinterColumns": [
                        {"name": "Model", "type": "string", "jsonPath": ".spec.model"},
                        {"name": "Backend", "type": "string", "jsonPath": ".spec.backend"},
                        {"name": "Replicas", "type": "integer", "jsonPath": ".spec.replicas"},
                        {"name": "Ready", "type": "integer", "jsonPath": ".status.readyReplicas"},
                        {"name": "Canary", "type": "string", "jsonPath": ".status.canaryState"},
                        {
                            "name": "Age",
                            "type": "date",
                            "jsonPath": ".metadata.creationTimestamp",
                        },
                    ],
                    "schema": {
                        "openAPIV3Schema": {
                            "type": "object",
                            "required": ["spec"],
                            "properties": {
                                "spec": {
                                    "type": "object",
                                    "required": ["model"],
                                    "properties": {
                                        "model": {"type": "string", "minLength": 1},
                                        "image": {"type": "string"},
                                        "replicas": {"type": "integer", "minimum": 0, "maximum": 100},
                                        "port": {"type": "integer", "minimum": 1, "maximum": 65535},
                                        "backend": {
                                            "type": "string",
                                            "enum": ["echo", "transformers", "vllm"],
                                        },
                                        "quantization": {
                                            "type": "string",
                                            "enum": ["awq", "gptq", "fp8", "none"],
                                        },
                                        "maxModelLen": {"type": "integer", "minimum": 128},
                                        "gpuMemoryUtilization": {
                                            "type": "number",
                                            "minimum": 0.1,
                                            "maximum": 1.0,
                                        },
                                        "maxBatchSize": {"type": "integer", "minimum": 1},
                                        "maxBatchTokens": {"type": "integer", "minimum": 1},
                                        "maxWaitMs": {"type": "number", "minimum": 0},
                                        "resources": {
                                            "type": "object",
                                            "properties": {
                                                "cpuRequest": {"type": "string"},
                                                "memoryRequest": {"type": "string"},
                                                "cpuLimit": {"type": "string"},
                                                "memoryLimit": {"type": "string"},
                                                "gpu": {"type": "integer", "minimum": 0},
                                            },
                                        },
                                        "autoscaling": {
                                            "type": "object",
                                            "properties": {
                                                "enabled": {"type": "boolean"},
                                                "minReplicas": {"type": "integer", "minimum": 0},
                                                "maxReplicas": {"type": "integer", "minimum": 1},
                                                "targetQueueDepth": {"type": "integer", "minimum": 1},
                                                "targetKvCacheUtilization": {
                                                    "type": "number",
                                                    "minimum": 0.1,
                                                    "maximum": 1.0,
                                                },
                                                "scaleDownStabilizationS": {"type": "integer", "minimum": 0},
                                                "metrics": {
                                                    "type": "array",
                                                    "items": {
                                                        "type": "string",
                                                        "enum": ["queue_depth", "kv_cache", "cpu"],
                                                    },
                                                },
                                            },
                                        },
                                        "canary": {
                                            "type": "object",
                                            "properties": {
                                                "enabled": {"type": "boolean"},
                                                "initialWeight": {
                                                    "type": "integer",
                                                    "minimum": 1,
                                                    "maximum": 100,
                                                },
                                                "step": {"type": "integer", "minimum": 1, "maximum": 100},
                                                "maxErrorRate": {
                                                    "type": "number",
                                                    "minimum": 0,
                                                    "maximum": 1,
                                                },
                                                "minSamples": {"type": "integer", "minimum": 1},
                                                "analysisIntervalS": {"type": "integer", "minimum": 5},
                                            },
                                        },
                                        "env": {"type": "object", "additionalProperties": {"type": "string"}},
                                        "nodeSelector": {
                                            "type": "object",
                                            "additionalProperties": {"type": "string"},
                                        },
                                        "serviceMonitor": {"type": "boolean"},
                                    },
                                },
                                "status": {
                                    "type": "object",
                                    "properties": {
                                        "observedGeneration": {"type": "integer"},
                                        "replicas": {"type": "integer"},
                                        "readyReplicas": {"type": "integer"},
                                        "canaryWeight": {"type": "integer"},
                                        "canaryState": {"type": "string"},
                                        "endpoint": {"type": "string"},
                                        "conditions": {"type": "array", "items": {"type": "object"}},
                                    },
                                },
                            },
                        }
                    },
                }
            ],
        },
    }


# --------------------------------------------------------------------- 解析
def _pick(data: dict[str, Any], *names: str, default: Any = None) -> Any:
    """从 dict 中按多个候选键名取值（兼容 camelCase / snake_case）。"""
    for n in names:
        if n in data and data[n] is not None:
            return data[n]
    return default


def parse_spec(raw: dict[str, Any]) -> InferenceServiceSpec:
    """把 CR 的 ``spec``（camelCase）解析为领域模型。

    宽容解析：未知字段忽略、缺失字段用默认值 —— 这样 CRD 演进时
    旧版本 CR 仍能被新版本 Operator 处理（Kubernetes API 兼容性要求）。
    """
    res_raw = _pick(raw, "resources", default={}) or {}
    auto_raw = _pick(raw, "autoscaling", default={}) or {}
    canary_raw = _pick(raw, "canary", default={}) or {}

    resources = ResourceRequirements(
        cpu_request=_pick(res_raw, "cpuRequest", "cpu_request", default="500m"),
        memory_request=_pick(res_raw, "memoryRequest", "memory_request", default="512Mi"),
        cpu_limit=_pick(res_raw, "cpuLimit", "cpu_limit", default="2000m"),
        memory_limit=_pick(res_raw, "memoryLimit", "memory_limit", default="2Gi"),
        gpu=int(_pick(res_raw, "gpu", default=0)),
    )
    autoscaling = AutoscalingSpec(
        enabled=bool(_pick(auto_raw, "enabled", default=True)),
        min_replicas=int(_pick(auto_raw, "minReplicas", "min_replicas", default=1)),
        max_replicas=int(_pick(auto_raw, "maxReplicas", "max_replicas", default=5)),
        target_queue_depth=int(_pick(auto_raw, "targetQueueDepth", "target_queue_depth", default=16)),
        target_kv_cache_utilization=float(
            _pick(auto_raw, "targetKvCacheUtilization", "target_kv_cache_utilization", default=0.8)
        ),
        scale_down_stabilization_s=int(
            _pick(auto_raw, "scaleDownStabilizationS", "scale_down_stabilization_s", default=300)
        ),
        metrics=list(_pick(auto_raw, "metrics", default=["queue_depth"])),
    )
    canary = CanarySpec(
        enabled=bool(_pick(canary_raw, "enabled", default=False)),
        initial_weight=int(_pick(canary_raw, "initialWeight", "initial_weight", default=10)),
        step=int(_pick(canary_raw, "step", default=20)),
        max_error_rate=float(_pick(canary_raw, "maxErrorRate", "max_error_rate", default=0.05)),
        min_samples=int(_pick(canary_raw, "minSamples", "min_samples", default=100)),
        analysis_interval_s=int(_pick(canary_raw, "analysisIntervalS", "analysis_interval_s", default=60)),
    )

    return InferenceServiceSpec(
        model=str(_pick(raw, "model", default="unknown-model")),
        image=str(_pick(raw, "image", default=DEFAULT_IMAGE)),
        replicas=int(_pick(raw, "replicas", default=1)),
        port=int(_pick(raw, "port", default=DEFAULT_PORT)),
        backend=str(_pick(raw, "backend", default="vllm")),
        quantization=_pick(raw, "quantization", default=None),
        max_model_len=int(_pick(raw, "maxModelLen", "max_model_len", default=4096)),
        gpu_memory_utilization=float(
            _pick(raw, "gpuMemoryUtilization", "gpu_memory_utilization", default=0.85)
        ),
        max_batch_size=int(_pick(raw, "maxBatchSize", "max_batch_size", default=16)),
        max_batch_tokens=int(_pick(raw, "maxBatchTokens", "max_batch_tokens", default=8192)),
        max_wait_ms=float(_pick(raw, "maxWaitMs", "max_wait_ms", default=20.0)),
        resources=resources,
        autoscaling=autoscaling,
        canary=canary,
        env=dict(_pick(raw, "env", default={}) or {}),
        node_selector=dict(_pick(raw, "nodeSelector", "node_selector", default={}) or {}),
        service_monitor=bool(_pick(raw, "serviceMonitor", "service_monitor", default=True)),
    )


def validate_spec(spec: InferenceServiceSpec) -> list[str]:
    """语义校验（schema 管不到的跨字段约束）。

    Returns:
        错误信息列表；空列表表示通过。
    """
    errors: list[str] = []
    if not spec.model.strip():
        errors.append("spec.model 不能为空")
    if spec.replicas < 0:
        errors.append("spec.replicas 不能为负")
    if spec.resources.gpu > 0 and spec.backend == "echo":
        errors.append("backend=echo 是确定性假后端，不应申请 GPU")
    if spec.autoscaling.enabled and spec.autoscaling.max_replicas < spec.autoscaling.min_replicas:
        errors.append("autoscaling.maxReplicas 必须 >= minReplicas")
    if spec.autoscaling.enabled and spec.autoscaling.max_replicas < spec.replicas:
        errors.append("autoscaling.maxReplicas 必须 >= spec.replicas")
    if spec.canary.enabled and spec.replicas < 2 and not spec.autoscaling.enabled:
        errors.append("启用灰度发布时 replicas 应 >= 2，否则无法共存新旧版本")
    if spec.canary.initial_weight > 100 or spec.canary.step > 100:
        errors.append("canary 权重与步长取值应在 1..100")
    if not 0.0 <= spec.gpu_memory_utilization <= 1.0:
        errors.append("gpuMemoryUtilization 应在 (0,1]")
    if spec.max_wait_ms < 0:
        errors.append("maxWaitMs 不能为负")
    return errors
