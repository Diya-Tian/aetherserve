"""配置系统测试：环境变量解析、前缀隔离、以及一个真实 bug 的回归保护。

这里有一个**必须记住的教训**
------------------------
最初 ``Settings`` 继承 ``BaseSettings`` 且带 ``env_prefix="AETHERSERVE_"``，
而它有一个名为 ``backend`` 的字段指向嵌套的 ``BackendSettings``，
后者**同样**带 ``AETHERSERVE_`` 前缀。于是：

    AETHERSERVE_BACKEND=echo   →  被父类当成给"复杂类型字段 backend"赋值
                               →  尝试把字符串 "echo" 当 JSON 解析
                               →  JSONDecodeError，**服务启动直接崩溃**

为什么本地开发发现不了：开发者一般不设环境变量，用的是代码默认值。
为什么测试也发现不了：单元测试直接构造 ``Settings()``，同样不读环境变量。
**最先抓到它的是 CI 的镜像 smoke test**（容器里通过环境变量注入配置）——
这与生产部署方式一致。

修法：聚合类改用 ``BaseModel``（不参与环境变量解析），
子配置块在 ``from_env()`` 里显式构造。
本文件用**子进程**跑回归测试：环境变量必须在解释器启动前设置，
进程内 ``os.environ`` 修改对已构造的 pydantic 对象不可靠。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from aetherserve.config import (
    BackendSettings,
    BatchingSettings,
    CircuitBreakerSettings,
    RateLimitSettings,
    Settings,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_settings_in_subprocess(env: dict[str, str]) -> tuple[bool, str]:
    """在干净的子进程里加载配置，返回 (是否成功, 输出)。

    清空继承的 ``AETHERSERVE_*`` 变量，保证测试只受传入的 env 影响。
    """
    import os

    child_env = {k: v for k, v in os.environ.items() if not k.startswith("AETHERSERVE_")}
    child_env.update(env)
    child_env["PYTHONPATH"] = str(REPO_ROOT / "src")
    child_env["PYTHONIOENCODING"] = "utf-8"

    probe = (
        "from aetherserve.config import get_settings;"
        "s = get_settings();"
        "print('RESULT', s.backend.backend, s.backend.model, s.batching.max_batch_size,"
        " s.ratelimit.enabled, s.circuit_breaker.failure_threshold)"
    )
    p = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=child_env,
        timeout=120,
    )
    output = (p.stdout or "") + (p.stderr or "")
    return p.returncode == 0, output


class TestEnvVarRegression:
    """回归保护：每个受支持的环境变量都必须能单独生效且不崩溃。"""

    @pytest.mark.parametrize(
        "env,expect_in_output",
        [
            ({"AETHERSERVE_BACKEND": "echo"}, "echo"),
            ({"AETHERSERVE_MODEL": "Qwen/Qwen2.5-1.5B-Instruct"}, "Qwen"),
            ({"AETHERSERVE_BATCH_MAX_BATCH_SIZE": "32"}, "32"),
            ({"AETHERSERVE_RATELIMIT_ENABLED": "false"}, "False"),
            ({"AETHERSERVE_CB_FAILURE_THRESHOLD": "9"}, "9"),
        ],
    )
    def test_single_env_var_does_not_crash(self, env: dict[str, str], expect_in_output: str) -> None:
        """这是那个崩溃 bug 的回归用例。

        特别是 ``AETHERSERVE_BACKEND``：修复前它会抛 JSONDecodeError。
        """
        ok, output = load_settings_in_subprocess(env)
        assert ok, f"设置 {env} 后配置加载失败（疑似环境变量名冲突）：\n{output}"
        assert expect_in_output in output, f"环境变量 {env} 未生效，输出：{output}"

    def test_all_supported_env_vars_together(self) -> None:
        """模拟生产部署：一次性注入全部配置（K8s 就是这么干的）。"""
        env = {
            "AETHERSERVE_BACKEND": "vllm",
            "AETHERSERVE_MODEL": "Qwen/Qwen2.5-1.5B-Instruct-AWQ",
            "AETHERSERVE_QUANTIZATION": "awq",
            "AETHERSERVE_MAX_MODEL_LEN": "4096",
            "AETHERSERVE_GPU_MEMORY_UTILIZATION": "0.85",
            "AETHERSERVE_BATCH_MAX_BATCH_SIZE": "24",
            "AETHERSERVE_BATCH_MAX_WAIT_MS": "15.0",
            "AETHERSERVE_RATELIMIT_ENABLED": "true",
            "AETHERSERVE_CB_FAILURE_THRESHOLD": "3",
            "AETHERSERVE_ENVIRONMENT": "prod",
            "AETHERSERVE_LOG_LEVEL": "INFO",
            "AETHERSERVE_API_KEYS": "sk-live:tenant-live:50:100",
            "AETHERSERVE_VLLM_ENDPOINT": "http://vllm:8001",
            "AETHERSERVE_PORT": "8080",
        }
        ok, output = load_settings_in_subprocess(env)
        assert ok, f"生产形态的全量环境变量注入导致加载失败：\n{output}"
        assert "vllm" in output
        assert "24" in output
        assert "3" in output

    def test_no_env_vars_uses_defaults(self) -> None:
        """零环境变量时必须用代码默认值（本地开发路径）。"""
        ok, output = load_settings_in_subprocess({})
        assert ok, output
        assert "echo" in output, "默认后端应为 echo"


class TestPrefixIsolation:
    """各子配置块的前缀必须互不干扰。"""

    def test_each_block_has_own_prefix(self) -> None:
        assert BatchingSettings.model_config["env_prefix"] == "AETHERSERVE_BATCH_"
        assert RateLimitSettings.model_config["env_prefix"] == "AETHERSERVE_RATELIMIT_"
        assert BackendSettings.model_config["env_prefix"] == "AETHERSERVE_"
        assert CircuitBreakerSettings.model_config["env_prefix"] == "AETHERSERVE_CB_"
        # 前缀两两之间不能有"一个是另一个的前缀又同名"的关系（BATCH_/RATELIMIT_/CB_ 各自独立）
        prefixes = [
            BatchingSettings.model_config["env_prefix"],
            RateLimitSettings.model_config["env_prefix"],
            CircuitBreakerSettings.model_config["env_prefix"],
        ]
        assert len(set(prefixes)) == len(prefixes), "子配置块前缀重复会导致配置互相覆盖"

    def test_aggregate_class_does_not_read_env(self) -> None:
        """聚合类必须不参与环境变量解析，否则会与子配置块冲突。

        这是修复 bug 的核心机制：如果 Settings 又是 BaseSettings，
        它就会去解释 AETHERSERVE_BACKEND 并崩掉。
        """
        from pydantic_settings import BaseSettings

        assert not issubclass(Settings, BaseSettings), (
            "Settings 不应继承 BaseSettings：会与嵌套的 BackendSettings 争夺 "
            "AETHERSERVE_BACKEND 这个变量名，导致把字符串当 JSON 解析而崩溃"
        )

    def test_sub_blocks_are_base_settings(self) -> None:
        from pydantic_settings import BaseSettings

        for cls in (BatchingSettings, RateLimitSettings, BackendSettings, CircuitBreakerSettings):
            assert issubclass(cls, BaseSettings), f"{cls.__name__} 应继承 BaseSettings 以读取自己的环境变量"


class TestDefaultsAndValidation:
    def test_from_env_returns_fully_populated_settings(self) -> None:
        s = Settings.from_env()
        assert isinstance(s.batching, BatchingSettings)
        assert isinstance(s.ratelimit, RateLimitSettings)
        assert isinstance(s.backend, BackendSettings)
        assert isinstance(s.circuit_breaker, CircuitBreakerSettings)

    def test_get_settings_is_cached(self) -> None:
        from aetherserve.config import get_settings

        assert get_settings() is get_settings(), "进程内应为单例，避免每请求重复解析"

    def test_default_backend_is_echo(self) -> None:
        """默认必须是无需 GPU 的 echo，保证 clone 下来就能跑起来。"""
        assert BackendSettings().backend == "echo"

    def test_api_keys_parsing(self) -> None:
        s = Settings.from_env()
        s.backend.api_keys = "sk-a:tenant-a:10:20,sk-b:tenant-b:30:40"
        parsed = s.parsed_api_keys()
        assert parsed["sk-a"] == ("tenant-a", 10.0, 20)
        assert parsed["sk-b"] == ("tenant-b", 30.0, 40)

    def test_api_keys_parsing_uses_ratelimit_defaults_when_omitted(self) -> None:
        s = Settings.from_env()
        s.backend.api_keys = "sk-only"
        parsed = s.parsed_api_keys()
        key, (tenant, rps, burst) = next(iter(parsed.items()))
        assert key == "sk-only"
        assert tenant == "sk-only", "未指定租户时应用 key 自身作为租户"
        assert rps == s.ratelimit.default_rps
        assert burst == s.ratelimit.default_burst

    def test_api_keys_parsing_skips_blanks(self) -> None:
        s = Settings.from_env()
        s.backend.api_keys = "sk-a:: , ,sk-b"
        parsed = s.parsed_api_keys()
        assert set(parsed) == {"sk-a", "sk-b"}

    def test_unknown_env_var_is_ignored(self) -> None:
        """extra="ignore"：未知环境变量不应导致启动失败（部署环境常有无关变量）。"""
        ok, output = load_settings_in_subprocess({"AETHERSERVE_TOTALLY_UNKNOWN": "1"})
        assert ok, output

    def test_env_prefix_constant_consistency(self) -> None:
        """环境变量前缀是运维契约，不能在代码里随意改动。"""
        # 这些前缀被 README/部署文档引用，改动会使线上配置静默失效
        documented = {
            "AETHERSERVE_": BackendSettings,
            "AETHERSERVE_BATCH_": BatchingSettings,
            "AETHERSERVE_RATELIMIT_": RateLimitSettings,
            "AETHERSERVE_CB_": CircuitBreakerSettings,
        }
        for prefix, cls in documented.items():
            assert cls.model_config["env_prefix"] == prefix, (
                f"{cls.__name__} 的 env_prefix 被改动，会破坏已有部署"
            )
