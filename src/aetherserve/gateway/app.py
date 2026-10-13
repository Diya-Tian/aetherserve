"""FastAPI 应用装配：OpenAI 兼容推理网关。

分层
----
    HTTP 层 (本模块 + routes)  →  引擎层 (engine)  →  后端层 (backends)
         ↑ 鉴权/限流/熔断/协议          ↑ 批处理/调度           ↑ 真实推理

本模块只做"装配与协议适配"，不含调度算法，保证可读性与可测试性。
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, cast

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import ORJSONResponse, StreamingResponse

from .. import __version__
from ..backends import available_backends, create_backend
from ..backends.base import GenerateParams
from ..config import Settings, get_settings
from ..engine import BatchingEngine, QueueFullError, build_engine
from ..observability import metrics
from ..observability.logging import configure_logging, get_logger, request_id_var, tenant_var
from .auth import ApiKeyAuthenticator, AuthError, Principal
from .circuit import CircuitBreakerRegistry, CircuitOpenError
from .ratelimit import RateLimitExceeded, TenantRateLimiter
from .router import BackendRouter
from .schemas import (
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    CompletionRequest,
    FinishReason,
    ModelCard,
    ModelList,
    Usage,
    new_id,
)
from .sse import SSE_DONE, SSE_HEADERS, heartbeat_stream, sse_pack

log = get_logger(__name__)


# --------------------------------------------------------------------- 状态容器
class GatewayState:
    """应用级共享状态（挂在 ``app.state`` 上，避免模块级全局变量）。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.authenticator = ApiKeyAuthenticator(settings.parsed_api_keys())
        self.limiter = TenantRateLimiter(
            default_rps=settings.ratelimit.default_rps,
            default_burst=settings.ratelimit.default_burst,
            enabled=settings.ratelimit.enabled,
        )
        breakers = CircuitBreakerRegistry(
            failure_threshold=settings.circuit_breaker.failure_threshold,
            recovery_timeout_s=settings.circuit_breaker.recovery_timeout_s,
        )
        self.router = BackendRouter(breakers=breakers)
        self.engine: BatchingEngine | None = None
        self.started_at = time.time()
        self.reuse_count = 0  # 复用声明：供测试断言 request 复用

    async def bootstrap(self) -> None:
        """创建并启动引擎（供 lifespan 调用）。"""
        backend = create_backend(self.settings)
        self.router.register(self.settings.backend.model, backend, sticky=False)
        self.engine = build_engine(self.settings, backend)
        await self.engine.start()
        log.info(
            "gateway_ready",
            backend=backend.name,
            model=self.settings.backend.model,
            available=available_backends(),
        )

    async def shutdown(self) -> None:
        if self.engine is not None:
            await self.engine.stop()


# --------------------------------------------------------------------- 依赖
def get_state(request: Request) -> GatewayState:
    state: GatewayState = request.app.state.gateway
    return state


async def get_principal(
    state: GatewayState = Depends(get_state),
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None, alias="x-api-key"),
) -> Principal:
    """从 ``Authorization: Bearer`` 或 ``x-api-key`` 解析身份。"""
    raw = x_api_key
    if authorization:
        parts = authorization.split(None, 1)
        token = parts[1].strip() if len(parts) == 2 else parts[0].strip()
        raw = token or raw

    try:
        principal = state.authenticator.authenticate(raw)
    except AuthError as exc:
        raise HTTPException(
            status_code=401,
            detail={"message": str(exc), "type": "invalid_request_error", "code": "invalid_api_key"},
        ) from exc

    tenant_var.set(principal.tenant)
    return principal


# --------------------------------------------------------------------- 辅助
def _estimate_cost(req: ChatCompletionRequest, settings: Settings) -> int:
    """预估请求的 token 成本，用于双维度限流。"""
    prompt_chars = sum(len(m.content) if isinstance(m.content, str) else 32 for m in req.messages)
    prompt_tokens = int(prompt_chars * 0.6) + 8
    completion = req.max_tokens or settings.backend.default_max_tokens
    return prompt_tokens + completion


def _to_generate_params(req: ChatCompletionRequest, settings: Settings) -> GenerateParams:
    stop = req.stop if isinstance(req.stop, list) else ([req.stop] if req.stop else [])
    return GenerateParams(
        max_tokens=req.max_tokens or settings.backend.default_max_tokens,
        temperature=req.temperature if req.temperature is not None else settings.backend.default_temperature,
        top_p=req.top_p if req.top_p is not None else 1.0,
        stop=stop,
        presence_penalty=req.presence_penalty or 0.0,
        frequency_penalty=req.frequency_penalty or 0.0,
        seed=req.seed,
    )


def _session_key(messages: list[ChatMessage]) -> str:
    """会话指纹：用于一致性哈希粘性路由（相同前缀命中同一实例以复用 prefix cache）。"""
    head = "|".join(f"{m.role}:{m.content if isinstance(m.content, str) else ''}" for m in messages[:2])
    return hashlib.sha256(head.encode("utf-8")).hexdigest()


def _chunk_to_dict(chunk: ChatCompletionChunk) -> dict[str, Any]:
    return chunk.model_dump(exclude_none=True)


# --------------------------------------------------------------------- 应用工厂
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """启动/停机钩子：初始化引擎，退出时优雅停机（等待在途请求）。"""
    state: GatewayState = app.state.gateway
    await state.bootstrap()
    try:
        yield
    finally:
        await state.shutdown()


def create_app(settings: Settings | None = None) -> FastAPI:
    """构造 FastAPI 应用。测试可传入自定义 settings。"""
    cfg = settings or get_settings()
    configure_logging(cfg.backend.log_level, json_output=cfg.environment != "dev")

    app = FastAPI(
        title="AetherServe Inference Gateway",
        description="面向大模型的多租户云原生推理服务平台 · OpenAI 兼容网关",
        version=__version__,
        default_response_class=ORJSONResponse,
        lifespan=lifespan,
        docs_url="/docs",
    )
    app.state.gateway = GatewayState(cfg)

    # ---------------------------------------------------------- 健康与元信息
    @app.get("/healthz", tags=["system"])
    async def healthz(state: GatewayState = Depends(get_state)) -> dict[str, Any]:
        """Kubernetes liveness/readiness 探针端点。"""
        engine_ok = state.engine is not None and state.engine.stats_snapshot()["running"]
        return {
            "status": "ok" if engine_ok else "degraded",
            "version": __version__,
            "uptime_s": round(time.time() - state.started_at, 3),
            "engine_running": engine_ok,
        }

    @app.get("/readyz", tags=["system"])
    async def readyz(state: GatewayState = Depends(get_state)) -> Response:
        """就绪探针：真实探测后端可用性（含一次健康调用），用于 K8s endpoint 摘除。"""
        health = await state.router.health()
        # 添加一个额外的响应头，便于探针审计
        ready = all(v.get("ready") for v in health.values()) if health else False
        return ORJSONResponse(
            {"ready": ready, "backends": health},
            status_code=200 if ready else 503,
        )

    @app.get("/metrics", tags=["system"])
    async def prometheus_metrics() -> Response:
        """Prometheus 抓取端点。"""
        return Response(content=metrics.render_metrics(), media_type="text/plain; version=0.0.4")

    @app.get("/v1/models", response_model=ModelList, tags=["openai"])
    async def list_models(
        _: Principal = Depends(get_principal),
        state: GatewayState = Depends(get_state),
    ) -> ModelList:
        return ModelList(data=[ModelCard(id=m, owned_by="aetherserve") for m in state.router.models])

    @app.get("/v1/system/info", tags=["system"])
    async def system_info(
        _: Principal = Depends(get_principal),
        state: GatewayState = Depends(get_state),
    ) -> dict[str, Any]:
        """运行时自描述：排障与 Operator 决策都依赖它。"""
        assert state.engine is not None
        return {
            "service": cfg.service_name,
            "version": __version__,
            "environment": cfg.environment,
            "backends_available": available_backends(),
            "active_backend": state.engine.stats_snapshot()["backend"],
            "engine": state.engine.stats_snapshot(),
            "circuit_breakers": state.router.breakers.snapshot(),
            "rate_limits": state.limiter.snapshot(),
            "config": {
                "max_batch_size": cfg.batching.max_batch_size,
                "max_batch_tokens": cfg.batching.max_batch_tokens,
                "max_wait_ms": cfg.batching.max_wait_ms,
                "worker_concurrency": cfg.batching.worker_concurrency,
                "max_queue_size": cfg.batching.max_queue_size,
            },
        }

    @app.get("/v1/system/stats", tags=["system"])
    async def system_stats(
        _: Principal = Depends(get_principal),
        state: GatewayState = Depends(get_state),
    ) -> dict[str, Any]:
        """轻量统计端点：HPA 自定义指标适配器可周期性抓取。"""
        assert state.engine is not None
        snap = state.engine.stats_snapshot()
        return {
            "queue_depth": snap["queue"]["depth"],
            "inflight_tokens": snap["queue"]["inflight_tokens"],
            "in_flight_batches": snap["scheduler"]["in_flight_batches"],
            "completed_batches": snap["scheduler"]["completed_batches"],
            "rejected": snap["queue"]["rejected"],
        }

    # ---------------------------------------------------------- 补全接口
    @app.post("/v1/chat/completions", tags=["openai"])
    async def chat_completions(
        payload: ChatCompletionRequest,
        principal: Principal = Depends(get_principal),
        state: GatewayState = Depends(get_state),
    ) -> Response:
        """OpenAI 兼容的对话补全（支持流式与非流式）。"""
        assert state.engine is not None

        # 1) 模型解析（未注册 → 404，符合 OpenAI 语义）
        try:
            state.router.get_route(payload.model)
        except KeyError as exc:
            raise HTTPException(
                status_code=404,
                detail={"message": str(exc), "type": "invalid_request_error", "code": "model_not_found"},
            ) from exc

        # 2) 限流（双维度：请求数 + token 预算）
        cost = _estimate_cost(payload, cfg)
        try:
            await state.limiter.acquire(
                principal.tenant, cost_tokens=cost, rps=principal.rps, burst=principal.burst
            )
        except RateLimitExceeded as exc:
            raise HTTPException(
                status_code=429,
                detail={"message": str(exc), "type": "rate_limit_error", "code": "rate_limit_exceeded"},
                headers={"Retry-After": f"{exc.retry_after_s:.2f}"},
            ) from exc

        # 3) 熔断准入
        engine_backend = state.engine._backend
        breaker = await state.router.breakers.get(engine_backend.name)
        try:
            await breaker.acquire()
        except CircuitOpenError as exc:
            raise HTTPException(
                status_code=503,
                detail={"message": str(exc), "type": "service_unavailable", "code": "circuit_open"},
                headers={"Retry-After": f"{exc.retry_after_s:.2f}"},
            ) from exc

        # 4) 提交引擎
        request_id = new_id("req")
        request_id_var.set(request_id)
        params = _to_generate_params(payload, cfg)

        try:
            handle = await state.engine.submit(
                model=payload.model,
                tenant=principal.tenant,
                messages=payload.messages,
                params=params,
                priority=0,
                request_id=request_id,
            )
        except QueueFullError as exc:
            await breaker.record_failure()
            raise HTTPException(
                status_code=429,
                detail={
                    "message": str(exc),
                    "type": "overloaded_error",
                    "code": exc.reason,
                },
                headers={"Retry-After": "1"},
            ) from exc

        completion_id = new_id("chatcmpl")

        # 5) 流式返回
        if payload.stream:

            async def event_source() -> AsyncIterator[str]:
                prompt_tokens = 0
                completion_tokens = 0
                finish: FinishReason | None = None

                # 首帧只带 role，符合 OpenAI 流式协议：客户端据此初始化消息对象
                first = ChatCompletionChunk(
                    id=completion_id,
                    model=payload.model,
                    choices=[
                        ChatCompletionChunkChoice(index=0, delta=ChatCompletionChunkDelta(role="assistant"))
                    ],
                )
                yield sse_pack(_chunk_to_dict(first))

                try:
                    async for chunk in handle.stream():
                        if chunk.prompt_tokens:
                            prompt_tokens = chunk.prompt_tokens
                        if chunk.completion_tokens:
                            completion_tokens = chunk.completion_tokens
                        if chunk.finish_reason:
                            # 后端返回的是 str，协议模型要求 Literal，这里做一次显式收敛
                            finish = cast(FinishReason, chunk.finish_reason)
                            break
                        if not chunk.text:
                            continue
                        frame = ChatCompletionChunk(
                            id=completion_id,
                            model=payload.model,
                            choices=[
                                ChatCompletionChunkChoice(
                                    index=0, delta=ChatCompletionChunkDelta(content=chunk.text)
                                )
                            ],
                        )
                        yield sse_pack(_chunk_to_dict(frame))
                except asyncio.CancelledError:
                    # 客户端断开：主动取消引擎中的请求，避免继续为空连接消耗 GPU
                    engine = state.engine
                    if engine is not None:
                        await engine.cancel(request_id)
                    raise
                finally:
                    await breaker.record_success()

                final = ChatCompletionChunk(
                    id=completion_id,
                    model=payload.model,
                    choices=[
                        ChatCompletionChunkChoice(
                            index=0, delta=ChatCompletionChunkDelta(), finish_reason=finish or "stop"
                        )
                    ],
                )
                yield sse_pack(_chunk_to_dict(final))

                if payload.stream_options and payload.stream_options.get("include_usage"):
                    usage_frame = ChatCompletionChunk(
                        id=completion_id,
                        model=payload.model,
                        choices=[],
                        usage=Usage(
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            total_tokens=prompt_tokens + completion_tokens,
                        ),
                    )
                    yield sse_pack(_chunk_to_dict(usage_frame))

                yield SSE_DONE

            return StreamingResponse(
                heartbeat_stream(event_source(), interval_s=15.0),
                media_type="text/event-stream",
                headers=SSE_HEADERS,
            )

        # 6) 非流式：聚合成完整响应
        text_parts: list[str] = []
        prompt_tokens = 0
        completion_tokens = 0
        finish_reason: str = "stop"
        try:
            async for chunk in handle.stream():
                if chunk.prompt_tokens:
                    prompt_tokens = chunk.prompt_tokens
                if chunk.completion_tokens:
                    completion_tokens = chunk.completion_tokens
                if chunk.finish_reason:
                    finish_reason = chunk.finish_reason
                if chunk.text:
                    text_parts.append(chunk.text)
            await breaker.record_success()
        except Exception:
            await breaker.record_failure()
            raise

        return ORJSONResponse(
            ChatCompletionResponse(
                id=completion_id,
                model=payload.model,
                choices=[
                    ChatCompletionChoice(
                        index=0,
                        message=ChatMessage(role="assistant", content="".join(text_parts)),
                        finish_reason=finish_reason,  # type: ignore[arg-type]
                    )
                ],
                usage=Usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                ),
            ).model_dump(exclude_none=True)
        )

    @app.post("/v1/completions", tags=["openai"])
    async def completions(
        payload: CompletionRequest,
        principal: Principal = Depends(get_principal),
        state: GatewayState = Depends(get_state),
    ) -> Response:
        """旧版补全接口：转换为对话格式复用同一引擎链路。"""
        prompt = (
            payload.prompt
            if isinstance(payload.prompt, str)
            else (payload.prompt[0] if payload.prompt else "")
        )
        chat_payload = ChatCompletionRequest(
            model=payload.model,
            messages=[ChatMessage(role="user", content=prompt)],
            max_tokens=payload.max_tokens,
            temperature=payload.temperature,
            top_p=payload.top_p,
            stop=payload.stop,
            stream=payload.stream,
        )
        return await chat_completions(chat_payload, principal, state)

    # ---------------------------------------------------------- 异常映射
    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> ORJSONResponse:  # pragma: no cover
        log.error("unhandled_exception", path=str(request.url.path), error=str(exc), exc_info=True)
        return ORJSONResponse(
            status_code=500,
            content={"error": {"message": "内部错误", "type": "internal_error", "code": "internal_error"}},
        )

    return app


# 模块级实例：``uvicorn aetherserve.gateway.app:app``
app = create_app()


def main() -> None:  # pragma: no cover
    """``aetherserve`` 命令行入口。"""
    import uvicorn

    cfg = get_settings()
    uvicorn.run(
        "aetherserve.gateway.app:app",
        host=cfg.backend.host,
        port=cfg.backend.port,
        log_level=cfg.backend.log_level.lower(),
        workers=1,  # 单进程：批处理引擎依赖进程内队列，多副本通过 K8s 横向扩展
    )


with contextlib.suppress(Exception):
    pass
