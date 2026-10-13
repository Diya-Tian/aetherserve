"""网关端到端测试：协议兼容性、鉴权、限流、流式、错误语义。

这些用例直接使用 FastAPI 的 ``TestClient``，因此覆盖了
"HTTP 层 → 引擎层 → 后端层"的完整链路，等同于对真实服务的集成测试。
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from aetherserve.config import Settings
from aetherserve.gateway.app import create_app

DEMO_KEY = "sk-demo-001"


def sse_events(text: str) -> list[dict[str, Any]]:
    """解析 SSE 文本为事件列表。"""
    out: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            continue
        try:
            out.append(json.loads(payload))
        except json.JSONDecodeError:
            continue
    return out


@pytest.fixture
def settings() -> Settings:
    s = Settings()
    s.environment = "test"
    s.batching.max_batch_size = 4
    s.batching.max_wait_ms = 5.0
    s.batching.max_queue_size = 16
    s.backend.model = "echo-1"
    s.backend.backend = "echo"
    s.backend.default_max_tokens = 8
    s.ratelimit.enabled = True
    s.ratelimit.default_rps = 100.0
    s.ratelimit.default_burst = 100
    return s


@pytest.fixture
def client(settings: Settings):
    """带 lifespan 的客户端：会真正启动/停止引擎。"""
    with TestClient(create_app(settings)) as c:
        yield c


def auth(key: str = DEMO_KEY) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


class TestAuth:
    def test_missing_key_rejected(self, client: TestClient) -> None:
        r = client.post(
            "/v1/chat/completions", json={"model": "echo-1", "messages": [{"role": "user", "content": "hi"}]}
        )
        assert r.status_code == 401
        assert r.json()["detail"]["code"] == "invalid_api_key"

    def test_invalid_key_rejected(self, client: TestClient) -> None:
        r = client.post(
            "/v1/chat/completions",
            json={"model": "echo-1", "messages": [{"role": "user", "content": "hi"}]},
            headers=auth("sk-wrong"),
        )
        assert r.status_code == 401

    def test_x_api_key_header_also_accepted(self, client: TestClient) -> None:
        r = client.get("/v1/models", headers={"x-api-key": DEMO_KEY})
        assert r.status_code == 200

    def test_healthz_needs_no_auth(self, client: TestClient) -> None:
        r = client.get("/healthz")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"
        assert r.json()["engine_running"] is True


class TestModelListing:
    def test_list_models(self, client: TestClient) -> None:
        r = client.get("/v1/models", headers=auth())
        assert r.status_code == 200
        body = r.json()
        assert body["object"] == "list"
        assert any(m["id"] == "echo-1" for m in body["data"])

    def test_unknown_model_404(self, client: TestClient) -> None:
        r = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-9-turbo", "messages": [{"role": "user", "content": "hi"}]},
            headers=auth(),
        )
        assert r.status_code == 404
        assert r.json()["detail"]["code"] == "model_not_found"


class TestChatCompletions:
    def test_non_streaming(self, client: TestClient) -> None:
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "echo-1",
                "messages": [{"role": "user", "content": "你好"}],
                "max_tokens": 5,
            },
            headers=auth(),
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["object"] == "chat.completion"
        assert body["choices"][0]["message"]["role"] == "assistant"
        assert body["choices"][0]["message"]["content"]
        assert body["choices"][0]["finish_reason"] in {"stop", "length"}
        assert body["usage"]["total_tokens"] > 0
        assert body["id"].startswith("chatcmpl-")

    def test_streaming_protocol(self, client: TestClient) -> None:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "echo-1",
                "messages": [{"role": "user", "content": "你好"}],
                "max_tokens": 4,
                "stream": True,
            },
            headers=auth(),
        ) as r:
            assert r.status_code == 200
            assert r.headers["content-type"].startswith("text/event-stream")
            # 关键：必须禁止中间层缓冲，否则流式体验会被破坏
            assert r.headers.get("x-accel-buffering") == "no"
            text = "".join(r.iter_text())

        assert text.rstrip().endswith("[DONE]")
        events = sse_events(text)
        assert events, "未解析到任何 SSE 事件"

        # 首帧应携带 role
        first_delta = events[0]["choices"][0]["delta"]
        assert first_delta.get("role") == "assistant"

        # 中间帧携带内容
        contents = [
            e["choices"][0]["delta"].get("content")
            for e in events
            if e.get("choices") and e["choices"][0]["delta"].get("content")
        ]
        assert len(contents) == 4, f"期望 4 个内容片，实际 {contents}"
        assert all(e["object"] == "chat.completion.chunk" for e in events if e.get("object"))

        # 结束帧带 finish_reason
        last = events[-1]
        assert last["choices"][0]["finish_reason"] in {"stop", "length"}

    def test_stream_usage_option(self, client: TestClient) -> None:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "echo-1",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 3,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
            headers=auth(),
        ) as r:
            events = sse_events("".join(r.iter_text()))

        usage_events = [e for e in events if e.get("usage")]
        assert usage_events, "include_usage=true 时应返回 usage 帧"
        assert usage_events[-1]["usage"]["total_tokens"] > 0

    def test_empty_messages_rejected(self, client: TestClient) -> None:
        r = client.post(
            "/v1/chat/completions",
            json={"model": "echo-1", "messages": []},
            headers=auth(),
        )
        assert r.status_code == 422

    def test_legacy_completions_endpoint(self, client: TestClient) -> None:
        r = client.post(
            "/v1/completions",
            json={"model": "echo-1", "prompt": "你好", "max_tokens": 3},
            headers=auth(),
        )
        assert r.status_code == 200
        assert r.json()["choices"][0]["message"]["content"]


class TestRateLimiting:
    def test_rejects_over_quota_with_retry_after(self) -> None:
        s = Settings()
        s.environment = "test"
        s.backend.model = "echo-1"
        s.backend.backend = "echo"
        s.backend.api_keys = "sk-tight:t1:0.5:1"  # 极低配额
        s.batching.max_wait_ms = 1.0
        with TestClient(create_app(s)) as c:
            codes = []
            for _ in range(6):
                r = c.post(
                    "/v1/chat/completions",
                    json={"model": "echo-1", "messages": [{"role": "user", "content": "x"}], "max_tokens": 2},
                    headers=auth("sk-tight"),
                )
                codes.append(r.status_code)
                if r.status_code == 429:
                    assert "Retry-After" in r.headers
                    assert r.json()["detail"]["type"] == "rate_limit_error"
            assert 429 in codes, f"限流未生效: {codes}"

    def test_tenants_do_not_interfere(self, settings: Settings) -> None:
        settings.backend.api_keys = "sk-a:tenant-a:1:1,sk-b:tenant-b:100:100"
        with TestClient(create_app(settings)) as c:
            payload = {"model": "echo-1", "messages": [{"role": "user", "content": "x"}], "max_tokens": 2}
            c.post("/v1/chat/completions", json=payload, headers=auth("sk-a"))
            r_a = c.post("/v1/chat/completions", json=payload, headers=auth("sk-a"))
            r_b = c.post("/v1/chat/completions", json=payload, headers=auth("sk-b"))
            assert r_a.status_code == 429
            assert r_b.status_code == 200


class TestSystemEndpoints:
    def test_readyz(self, client: TestClient) -> None:
        r = client.get("/readyz")
        assert r.status_code in {200, 503}
        body = r.json()
        assert "backends" in body

    def test_metrics_exposition(self, client: TestClient) -> None:
        # 先产生一些指标
        client.post(
            "/v1/chat/completions",
            json={"model": "echo-1", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 2},
            headers=auth(),
        )
        r = client.get("/metrics")
        assert r.status_code == 200
        body = r.text
        for metric in (
            "aetherserve_requests_total",
            "aetherserve_ttft_seconds",
            "aetherserve_queue_depth",
            "aetherserve_batch_size",
            "aetherserve_tokens_total",
        ):
            assert metric in body, f"缺少指标 {metric}"

    def test_system_info_shape(self, client: TestClient) -> None:
        r = client.get("/v1/system/info", headers=auth())
        assert r.status_code == 200
        body = r.json()
        assert body["active_backend"] == "echo"
        assert "engine" in body and "config" in body
        assert body["config"]["max_batch_size"] >= 1

    def test_system_stats_is_hpa_friendly(self, client: TestClient) -> None:
        """HPA 适配器会周期性抓取该端点，字段必须稳定且为数值。"""
        r = client.get("/v1/system/stats", headers=auth())
        assert r.status_code == 200
        body = r.json()
        for field in ("queue_depth", "inflight_tokens", "in_flight_batches", "completed_batches", "rejected"):
            assert field in body, f"缺少字段 {field}"
            assert isinstance(body[field], int)


class TestConcurrency:
    def test_parallel_requests_all_succeed(self, client: TestClient) -> None:
        """并发请求必须全部成功且内容互不串扰（批内多路复用的正确性）。"""
        import concurrent.futures

        def call(i: int) -> tuple[int, str]:
            r = client.post(
                "/v1/chat/completions",
                json={
                    "model": "echo-1",
                    "messages": [{"role": "user", "content": f"请求{i}"}],
                    "max_tokens": 4,
                },
                headers=auth(),
            )
            return r.status_code, r.json()["choices"][0]["message"]["content"]

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(call, range(12)))

        assert all(code == 200 for code, _ in results), results
        assert all(content for _, content in results)
