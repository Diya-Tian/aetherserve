# AetherServe 开发与运维命令集
#
# 说明：所有命令都刻意做成"单条可复制"，避免依赖某个特定 shell 的高级特性，
# 以便在 Windows PowerShell / WSL / CI 中都可用。

SHELL := /bin/sh
PY ?= python
VENV ?= .venv
PORT ?= 8000
IMAGE ?= ghcr.io/aetherserve/gateway:0.1.0
CLUSTER ?= aetherserve
NAMESPACE ?= ai-serving

.PHONY: help install dev test cov lint fmt typecheck bench bench-sweep verify e2e \
        docker-build cluster cluster-delete deploy undeploy operator crd clean all

help:  ## 显示所有可用命令
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ------------------------------------------------------------------ 开发
install:  ## 安装依赖（含开发与 Operator 依赖）
	$(PY) -m venv $(VENV)
	$(VENV)/bin/pip install -U pip
	$(VENV)/bin/pip install -e ".[dev,operator]"

dev:  ## 本地热重载启动网关（默认 Echo 后端，无需 GPU）
	$(VENV)/bin/uvicorn aetherserve.gateway.app:app --reload --host 0.0.0.0 --port $(PORT)

run:  ## 本地启动网关（不热重载，接近生产形态）
	$(VENV)/bin/uvicorn aetherserve.gateway.app:app --host 0.0.0.0 --port $(PORT) --workers 1

# ------------------------------------------------------------------ 质量
test:  ## 运行全部测试
	$(VENV)/bin/pytest tests/ -q

cov:  ## 运行测试并输出覆盖率
	$(VENV)/bin/pytest tests/ -q --cov=aetherserve --cov-report=term-missing

lint:  ## ruff 静态检查
	$(VENV)/bin/ruff check src tests bench

fmt:  ## ruff 自动修复 + 格式化
	$(VENV)/bin/ruff check --fix src tests bench
	$(VENV)/bin/ruff format src tests bench

typecheck:  ## mypy 类型检查
	$(VENV)/bin/mypy src

all: lint typecheck test  ## 一次跑完静态检查 + 类型检查 + 测试

# ------------------------------------------------------------------ 压测
bench:  ## 主对标压测：串行 vs 连续批处理
	$(VENV)/bin/python bench/run_bench.py --mode sim --concurrency 1,4,8,16,32 \
		--requests 64 --max-tokens 32 --prompt-tokens 256

bench-sweep:  ## 批大小扫描曲线
	$(VENV)/bin/python bench/run_bench.py --mode sweep --batch-sweep 1,2,4,8,16,32 \
		--sweep-concurrency 16 --requests 64 --max-tokens 32 --prompt-tokens 256

bench-echo:  ## 网关纯开销基线
	$(VENV)/bin/python bench/run_bench.py --mode echo --concurrency 1,16,64 --requests 96

bench-vllm:  ## 真实 GPU 压测（需先启动 vLLM）
	$(VENV)/bin/python bench/run_bench.py --mode vllm --endpoint $(VLLM_ENDPOINT) \
		--concurrency 1,8,16,32 --requests 64

verify:  ## 仿真模型自校验（解析预测 vs 实测，必须 PASS）
	$(VENV)/bin/python bench/verify_model.py

verify-claims:  ## 文档中的数字 vs 压测原始数据对账
	$(VENV)/bin/python bench/verify_claims.py

verify-chart:  ## 校验 Helm Chart（lint + 官方模型反序列化 + 安全不变量）
	helm lint deploy/helm/aetherserve
	$(VENV)/bin/python deploy/helm/verify_chart.py

verify-docs:  ## 文档一致性（README 链接、make 目标、脚本引用、孤儿文档）
	$(VENV)/bin/python tools/check_docs.py

verify-all: verify verify-claims verify-chart verify-docs  ## 跑完全部校验

render:  ## 渲染 Helm 清单到本地（便于 review）
	helm template aetherserve deploy/helm/aetherserve --namespace $(NAMESPACE) > deploy/helm/rendered.yaml
	@echo "已渲染到 deploy/helm/rendered.yaml"

# ------------------------------------------------------------------ 容器与集群
docker-build:  ## 构建网关镜像
	docker build -t $(IMAGE) .

cluster:  ## 创建本地 kind 集群并打标签（模拟 GPU 节点）
	kind create cluster --name $(CLUSTER) --config deploy/local/kind-cluster.yaml
	kubectl label node $(CLUSTER)-worker  aetherserve.io/accelerator=nvidia --overwrite
	kubectl create namespace $(NAMESPACE) --dry-run=client -o yaml | kubectl apply -f -

cluster-delete:  ## 删除本地集群
	kind delete cluster --name $(CLUSTER)

crd:  ## 从代码生成并安装 InferenceService CRD（代码即契约）
	$(VENV)/bin/python deploy/k8s/generate_manifests.py --all --out deploy/k8s/aetherserve-crd.yaml
	kubectl apply -f deploy/k8s/aetherserve-crd.yaml

deploy:  ## 用 Helm 部署平台（CRD + Operator + RBAC + 监控）
	helm upgrade --install aetherserve deploy/helm/aetherserve \
		--namespace $(NAMESPACE) --create-namespace --wait

undeploy:  ## 卸载平台
	helm uninstall aetherserve --namespace $(NAMESPACE) || true

operator:  ## 本地运行 Operator（读取 ~/.kube/config）
	$(VENV)/bin/python -m aetherserve.operator.main --verbose

e2e:  ## 在 kind 集群上跑端到端测试
	$(VENV)/bin/pytest tests/e2e -q -m e2e

# ------------------------------------------------------------------ 其他
manifest:  ## 渲染 Helm 清单到本地（便于 review）
	helm template aetherserve deploy/helm/aetherserve --namespace $(NAMESPACE) > /tmp/aetherserve-rendered.yaml
	@echo "已渲染到 /tmp/aetherserve-rendered.yaml"

clean:  ## 清理构建产物与缓存
	rm -rf $(VENV) .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage dist build *.egg-info
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
