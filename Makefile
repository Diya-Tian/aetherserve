# AetherServe 开发与运维命令集（make 入口）
#
# 关于跨平台
# ---------
# make 本身在 Windows 上默认不可用（需要 Git Bash / MSYS2 / WSL 提供 make 与 sh）。
# 因此提供两条等价路径：
#   · 有 make 的环境（Linux / WSL / CI）：直接 make <target>
#   · Windows PowerShell 或其他无 make 的环境：python tasks.py <task>
# 两者由同一套命令驱动，行为一致。
#
# 本文件已修正两个平台假设：
#   1. 自动探测 venv 可执行文件目录（Unix 是 bin/，Windows 是 Scripts/）；
#   2. clean 目标不依赖 rm -rf（Windows 无该命令），改用 Python 实现。

SHELL := /bin/sh
PY ?= python
VENV ?= .venv
PORT ?= 8000
IMAGE ?= aetherserve/gateway:0.1.0
CLUSTER ?= aetherserve
NAMESPACE ?= ai-serving

# 探测 venv 脚本目录：优先 bin/（Unix），否则 Scripts/（Windows）
VENV_BIN := $(firstword $(wildcard $(VENV)/bin $(VENV)/Scripts))
ifeq ($(VENV_BIN),)
VENV_BIN := $(VENV)/bin
endif

# 统一的解释器与工具入口
PYTHON := $(VENV_BIN)/python
PIP := $(VENV_BIN)/pip
PYTEST := $(VENV_BIN)/pytest
RUFF := $(VENV_BIN)/ruff
MYPY := $(VENV_BIN)/mypy
UVICORN := $(VENV_BIN)/uvicorn

.PHONY: help install dev run test cov lint fmt typecheck bench bench-sweep bench-echo \
        verify verify-claims verify-chart verify-docs verify-readme verify-coldstart verify-all \
        render docker-build cluster cluster-delete deploy undeploy operator crd e2e clean all

help:  ## 显示所有可用命令
	@echo "AetherServe 命令集（也可用 python tasks.py <task>）"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ------------------------------------------------------------------ 开发
install:  ## 安装依赖（含开发与 Operator 依赖）
	$(PY) -m venv $(VENV)
	$(PIP) install -U pip
	$(PIP) install -e ".[dev,operator]"

dev:  ## 本地热重载启动网关（默认 Echo 后端，无需 GPU）
	$(UVICORN) aetherserve.gateway.app:app --reload --host 0.0.0.0 --port $(PORT)

run:  ## 本地启动网关（不热重载，接近生产形态）
	$(UVICORN) aetherserve.gateway.app:app --host 0.0.0.0 --port $(PORT) --workers 1

# ------------------------------------------------------------------ 质量
test:  ## 运行全部测试（e2e 在无集群时自动跳过）
	$(PYTEST) tests/ -q

cov:  ## 运行测试并输出覆盖率
	$(PYTEST) tests/ -q --cov=aetherserve --cov-report=term-missing

lint:  ## ruff 静态检查
	$(RUFF) check src tests bench deploy tools

fmt:  ## ruff 自动修复 + 格式化
	$(RUFF) check --fix src tests bench deploy tools
	$(RUFF) format src tests bench deploy tools

typecheck:  ## mypy 类型检查
	$(MYPY) src

all: lint typecheck test  ## 一次跑完静态检查 + 类型检查 + 测试

# ------------------------------------------------------------------ 压测
bench:  ## 主对标压测：串行 vs 连续批处理
	$(PYTHON) bench/run_bench.py --mode sim --concurrency 1,4,8,16,32 \
		--requests 64 --max-tokens 32 --prompt-tokens 256

bench-sweep:  ## 批大小扫描曲线
	$(PYTHON) bench/run_bench.py --mode sweep --batch-sweep 1,2,4,8,16,32 \
		--sweep-concurrency 16 --requests 64 --max-tokens 32 --prompt-tokens 256

bench-echo:  ## 网关纯开销基线
	$(PYTHON) bench/run_bench.py --mode echo --concurrency 1,16,64 --requests 96

bench-vllm:  ## 真实 GPU 压测（需先启动 vLLM）
	$(PYTHON) bench/run_bench.py --mode vllm --endpoint $(VLLM_ENDPOINT) \
		--concurrency 1,8,16,32 --requests 64

verify:  ## 仿真模型自校验（解析预测 vs 实测，必须 PASS）
	$(PYTHON) bench/verify_model.py

verify-claims:  ## 文档中的数字 vs 压测原始数据对账
	$(PYTHON) bench/verify_claims.py

verify-chart:  ## 校验 Helm Chart（lint + 官方模型反序列化 + 安全不变量）
	helm lint deploy/helm/aetherserve
	$(PYTHON) deploy/helm/verify_chart.py

verify-docs:  ## 文档一致性（README 链接、make 目标、脚本引用、孤儿文档）
	$(PYTHON) tools/check_docs.py

verify-readme:  ## README 声明与仓库现实一致性（路径、链接、功能声称、隐私）
	$(PYTHON) tools/check_readme.py

verify-coldstart:  ## 冷启动端到端验证（全新 venv + 按 README 操作，24 项检查）
	$(PYTHON) tools/coldstart_check.py

verify-all: verify verify-claims verify-chart verify-docs verify-readme  ## 跑完全部校验（不含冷启动）

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
	$(PYTHON) deploy/k8s/generate_manifests.py --all --out deploy/k8s/aetherserve-crd.yaml
	kubectl apply -f deploy/k8s/aetherserve-crd.yaml

deploy:  ## 用 Helm 部署平台（CRD + Operator + RBAC + 监控）
	helm upgrade --install aetherserve deploy/helm/aetherserve \
		--namespace $(NAMESPACE) --create-namespace --wait

undeploy:  ## 卸载平台
	helm uninstall aetherserve --namespace $(NAMESPACE) || true

operator:  ## 本地运行 Operator（读取 ~/.kube/config）
	$(PYTHON) -m aetherserve.operator.main --verbose

e2e:  ## 在 kind 集群上跑端到端测试（无集群时自动跳过）
	$(PYTEST) tests/e2e -q -m e2e

# ------------------------------------------------------------------ 其他
clean:  ## 清理构建产物与缓存（跨平台实现，不依赖 rm -rf）
	$(PYTHON) tasks.py clean
