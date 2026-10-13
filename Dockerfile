# AetherServe 网关镜像（多阶段构建）
#
# 设计要点
# --------
# 1. **多阶段**：builder 阶段装编译依赖与 wheel，runtime 阶段只保留运行时依赖，
#    镜像体积显著缩小（避免把 gcc/rust 工具链带进生产镜像）；
# 2. **非 root 运行**：与 Operator 生成的 Pod securityContext 一致（uid 10001）；
# 3. **无 GPU 依赖**：网关本身不需要 CUDA，GPU 由独立的 vLLM 推理 Pod 承担。
#    这让网关镜像保持瘦小，可以快速横向扩容（HPA 冷启动时间更短）；
# 4. **HEALTHCHECK**：即使不经 K8s 探针（如 docker run）也能感知存活状态。

# ---------------------------------------------------------------- builder
FROM python:3.12-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

# 先装构建工具（只在 builder 阶段存在）
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# 先只复制依赖描述文件，利用 Docker 层缓存：源码改动不会导致重装依赖
COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install -U pip setuptools wheel \
    && /opt/venv/bin/pip install .

# ---------------------------------------------------------------- runtime
FROM python:3.12-slim AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # 默认用 Echo 后端启动（无 GPU 也能跑通健康检查与协议）
    AETHERSERVE_BACKEND=echo \
    AETHERSERVE_MODEL=echo-1 \
    AETHERSERVE_BACKEND_HOST=0.0.0.0 \
    AETHERSERVE_BACKEND_PORT=8000 \
    AETHERSERVE_ENVIRONMENT=prod

# curl 用于 HEALTHCHECK；tini 用于正确转发信号（优雅停机依赖它）
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl tini \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --uid 10001 --create-home --shell /usr/sbin/nologin aetherserve

COPY --from=builder /opt/venv /opt/venv

USER 10001

EXPOSE 8000

# 让 docker 直接运行时也有健康感知；K8s 场景由 readinessProbe/livenessProbe 接管
HEALTHCHECK --interval=30s --timeout=3s --start-period=15s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/healthz || exit 1

# tini 作为 PID 1：确保 SIGTERM 能传到 uvicorn，触发引擎的优雅停机（排空队列）
ENTRYPOINT ["/usr/bin/tini", "--"]

# workers=1 是刻意的：批处理队列是进程内状态，多副本通过 K8s 横向扩展而非多进程
CMD ["uvicorn", "aetherserve.gateway.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
