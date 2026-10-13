"""全局配置：全部支持 AETHERSERVE_ 前缀的环境变量覆盖。

设计说明
--------
配置集中在 `Settings` 中，便于在 Kubernetes 里通过 ConfigMap/环境变量注入，
避免把业务参数（批大小、超时、显存预算）硬编码在代码里 —— 这是云原生应用的基本要求。
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class BatchingSettings(BaseSettings):
    """连续批处理引擎参数（延迟与吞吐的核心旋钮）。"""

    model_config = SettingsConfigDict(env_prefix="AETHERSERVE_BATCH_", extra="ignore")

    max_batch_size: int = Field(default=16, ge=1, le=256, description="单批最大请求数")
    max_batch_tokens: int = Field(default=8192, ge=256, description="单批最大 token 总量")
    max_wait_ms: float = Field(default=20.0, ge=0.0, description="批构建最大等待时延，用于摊薄延迟")
    max_queue_size: int = Field(default=512, ge=1, description="准入队列容量，超出触发背压 429")
    max_queue_tokens: int = Field(default=32768, ge=1, description="在途请求预估 KV 总量上限（显存保护）")
    worker_concurrency: int = Field(default=4, ge=1, description="并发批执行器数量")
    bucket_step: int = Field(default=128, ge=1, description="prompt 长度分桶粒度，减少 padding 浪费")
    aging_ms: float = Field(default=500.0, ge=0.0, description="队列等待老化阈值，超过后提升优先级防饿死")


class RateLimitSettings(BaseSettings):
    """多租户限流参数（令牌桶）。"""

    model_config = SettingsConfigDict(env_prefix="AETHERSERVE_RATELIMIT_", extra="ignore")

    enabled: bool = True
    default_rps: float = Field(default=20.0, gt=0, description="默认租户每秒请求数")
    default_burst: int = Field(default=40, ge=1, description="默认租户突发容量")
    default_tpm: int = Field(default=200_000, gt=0, description="默认租户每分钟 token 配额")


class BackendSettings(BaseSettings):
    """推理后端选择与参数。"""

    model_config = SettingsConfigDict(env_prefix="AETHERSERVE_", extra="ignore")

    backend: str = Field(default="echo", description="echo | transformers | vllm")
    model: str = Field(default="echo-1", description="默认模型标识")
    model_path: str = Field(default="Qwen/Qwen2.5-1.5B-Instruct", description="本地/远端模型路径")
    quantization: str | None = Field(default=None, description="awq | gptq | None")

    # vLLM 相关
    gpu_memory_utilization: float = Field(default=0.85, gt=0, le=1.0)
    max_model_len: int = Field(default=4096, ge=128)
    vllm_endpoint: str | None = Field(default=None, description="远程 vLLM OpenAI server 地址")

    # 生成默认值
    default_max_tokens: int = Field(default=256, ge=1)
    default_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    request_timeout_s: float = Field(default=120.0, gt=0)

    # 服务
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    api_keys: str = Field(
        default="sk-demo-001:tenant-demo:20:40",
        description="API-Key 配置，格式 key:tenant:rps:burst，多个用逗号分隔",
    )


class CircuitBreakerSettings(BaseSettings):
    """熔断器参数：后端劣化时快速失败，避免请求堆积。"""

    model_config = SettingsConfigDict(env_prefix="AETHERSERVE_CB_", extra="ignore")

    failure_threshold: int = Field(default=5, ge=1, description="连续失败多少次后打开熔断")
    recovery_timeout_s: float = Field(default=10.0, gt=0, description="熔断打开后多久进入半开")
    half_open_max_calls: int = Field(default=2, ge=1, description="半开态允许的探测请求数")


class Settings(BaseSettings):
    """聚合配置入口。"""

    model_config = SettingsConfigDict(env_prefix="AETHERSERVE_", extra="ignore")

    service_name: str = "aetherserve"
    environment: str = Field(default="dev", description="dev | staging | prod")
    metrics_enabled: bool = True

    batching: BatchingSettings = Field(default_factory=BatchingSettings)
    ratelimit: RateLimitSettings = Field(default_factory=RateLimitSettings)
    backend: BackendSettings = Field(default_factory=BackendSettings)
    circuit_breaker: CircuitBreakerSettings = Field(default_factory=CircuitBreakerSettings)

    def parsed_api_keys(self) -> dict[str, tuple[str, float, int]]:
        """解析 API-Key 配置。

        Returns:
            {api_key: (tenant_id, rps, burst)}
        """
        result: dict[str, tuple[str, float, int]] = {}
        for raw in self.backend.api_keys.split(","):
            raw = raw.strip()
            if not raw:
                continue
            parts = raw.split(":")
            key = parts[0]
            tenant = parts[1] if len(parts) > 1 and parts[1] else key
            rps = float(parts[2]) if len(parts) > 2 and parts[2] else self.ratelimit.default_rps
            burst = int(parts[3]) if len(parts) > 3 and parts[3] else self.ratelimit.default_burst
            result[key] = (tenant, rps, burst)
        return result


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """进程内单例，避免每次请求重复解析环境变量。"""
    return Settings()
