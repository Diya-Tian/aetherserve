"""pytest 共享夹具。"""

from __future__ import annotations

import asyncio

import pytest

from aetherserve.backends.echo_backend import EchoBackend
from aetherserve.config import Settings
from aetherserve.engine import BatchPlan, Scheduler, build_engine
from aetherserve.engine.queue import AdmissionQueue


@pytest.fixture
def fast_settings() -> Settings:
    """把批处理参数调小，让测试在毫秒级完成，同时更快触发边界条件。"""
    s = Settings()
    s.batching.max_batch_size = 4
    s.batching.max_batch_tokens = 512
    s.batching.max_wait_ms = 5.0
    s.batching.max_queue_size = 16
    s.batching.max_queue_tokens = 2048
    s.batching.worker_concurrency = 2
    s.backend.model = "echo-1"
    s.backend.backend = "echo"
    s.backend.default_max_tokens = 16
    s.environment = "test"
    return s


@pytest.fixture
def echo_backend() -> EchoBackend:
    # 去掉人为延迟，测试只关心逻辑正确性
    return EchoBackend(token_delay_ms=0.0, max_tokens_hint=8)


@pytest.fixture
def plan() -> BatchPlan:
    return BatchPlan(max_batch_size=4, max_batch_tokens=512, max_wait_ms=5.0, bucket_step=64)


@pytest.fixture
def queue() -> AdmissionQueue:
    return AdmissionQueue(max_size=8, max_tokens=1024, aging_ms=100.0)


@pytest.fixture
async def engine(fast_settings: Settings, echo_backend: EchoBackend):
    """已启动的引擎，测试结束自动优雅停机。"""
    eng = build_engine(fast_settings, echo_backend)
    await eng.start()
    try:
        yield eng
    finally:
        await eng.stop(drain=False)


@pytest.fixture
def scheduler(plan: BatchPlan) -> Scheduler:
    return Scheduler(plan, concurrency=2, kv_budget_tokens=1024)


def pytest_configure(config: pytest.Config) -> None:
    """确保事件循环策略在 Windows 上使用 Selector（兼容性更好）。"""
    if hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):  # pragma: no cover - 仅 Windows
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
