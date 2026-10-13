# AetherServe — 面向大模型的多租户云原生推理服务平台

<p align="center">
  <b>OpenAI-compatible · Continuous Batching · Kubernetes Operator · GPU-aware Scheduling</b>
</p>

<p align="center">
  <a href="https://github.com/Diya-Tian/aetherserve/actions/workflows/ci.yml">
    <img alt="CI" src="https://github.com/Diya-Tian/aetherserve/actions/workflows/ci.yml/badge.svg">
  </a>
  <img alt="python" src="https://img.shields.io/badge/python-3.11%20%7C%203.12-blue">
  <img alt="kubernetes" src="https://img.shields.io/badge/kubernetes-1.29%2B-326CE5">
  <img alt="tests" src="https://img.shields.io/badge/tests-263%20passed-brightgreen">
  <img alt="coverage" src="https://img.shields.io/badge/coverage-86%25-green">
  <img alt="license" src="https://img.shields.io/badge/license-Apache--2.0-blue">
</p>

AetherServe 是一套面向大语言模型（LLM）的**多租户推理服务平台**。它对业务方暴露 OpenAI 兼容协议，
对内通过自研**连续批处理（Continuous Batching）调度引擎**提升 GPU 吞吐，通过 **Kubernetes Operator**
实现推理服务的声明式交付与自动扩缩容，通过 **GPU 感知调度器**提升显存装箱率。

> 面向场景：企业内部多团队共享 GPU 推理资源 —— 既要高吞吐，又要多租户隔离，还要能弹性伸缩。

---

## 核心特性

| 能力 | 实现 | 关键设计 |
|---|---|---|
| **协议兼容** | `/v1/chat/completions`、`/v1/completions`、`/v1/models` | OpenAI 协议子集 + SSE 流式，业务零改造迁移 |
| **连续批处理** | `src/aetherserve/engine/` | 优先级队列 + prompt 长度分桶 + 显存预算 + 逐 token 流式回传 |
| **多租户隔离** | `gateway/auth.py` + `gateway/ratelimit.py` | API-Key → 租户映射，令牌桶限流，批内公平调度 |
| **可插拔后端** | `backends/` | vLLM / Transformers / Echo 三实现，统一抽象接口 |
| **声明式交付** | `operator/` | CRD `InferenceService` + 调和循环自动生成 K8s 资源 |
| **弹性伸缩** | `operator/` + 自定义指标 | 队列深度 & KV Cache 水位驱动 HPA |
| **GPU 资源约束** | `operator/builders.py` | 通过 nodeSelector + tolerations 把 GPU 服务限定到 GPU 节点；显存超卖防护依赖准入预算（见 `engine/`） |
| **可观测性** | `observability/` | RED 指标 + LLM 专项指标（TTFT/TPOT/批大小/KV 水位） |
| **质量保障** | `tests/` + `.github/workflows/ci.yml` | 263 个单测 + ruff + mypy + 4 个自动化校验（模型/数字/Chart/文档） |

---

## 快速开始（无需 GPU）

Echo 后端是确定性的假推理后端，用于本地开发与 CI，**不依赖 GPU / 模型权重**。

```bash
# 1) 安装
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

# 2) 启动网关（默认 Echo 后端）
make dev            # 等价于: uvicorn aetherserve.gateway.app:app --reload --port 8000

# 3) 调用
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Authorization: Bearer sk-demo-001' \
  -H 'Content-Type: application/json' \
  -d '{"model":"echo-1","messages":[{"role":"user","content":"你好"}],"stream":true}'
```

预期输出（截断）：

```
data: {"id":"cmpl-...","object":"chat.completion.chunk","choices":[{"delta":{"content":"你"}}]}

data: {"id":"cmpl-...","object":"chat.completion.chunk","choices":[{"delta":{"content":"好"}}]}

data: [DONE]
```

### 使用真实模型（需要 NVIDIA GPU）

```bash
pip install -e ".[vllm]"
export AETHERSERVE_BACKEND=vllm
export AETHERSERVE_MODEL=Qwen/Qwen2.5-1.5B-Instruct-AWQ   # 8GB 显存可跑
make dev
```

---

## 性能数据

> 测试环境：Windows 11 / i7-13700H（14C20T）/ 16GB RAM / RTX 4060 Laptop 8GB
> 测试方式：`bench/run_bench.py --mode sim`，prompt 256 token、生成 32 token、每档 64 请求
>
> **重要声明**：下列数据来自**耗时模型仿真**（`src/aetherserve/sim/`），不是真实 GPU 实测。
> 仿真基于 Transformer 推理的物理事实——Prefill 受算力限制（线性）、Decode 受显存带宽限制
> （权重读取成本被批内摊薄），因此**相对关系**与真实引擎一致。
> 数据可信度由 `bench/verify_model.py` 独立校验：解析模型预测 vs 实测墙钟，**偏差 < 3%**（阈值 15%）。
> 并以 `bench/verify_claims.py` 自动对账"文档中的数字 vs 原始数据"，保证文档与报告可复现。

### 串行推理 vs 连续批处理（同并发窗口对比）

| 场景 | 并发 | QPS | TTFT P50 | TTFT P99 | E2E P99 | GPU 利用率 |
|---|---|---|---|---|---|---|
| 串行（无批处理） | 1 | 4.74 | 51.5ms | 70.4ms | 234.9ms | 5.8% |
| 串行（无批处理） | 16 | 4.63 | 3240.4ms | 3362.0ms | 3523.2ms | 5.8% |
| 串行（无批处理） | 32 | 4.64 | 6676.5ms | 6777.9ms | 6933.7ms | 5.8% |
| **连续批处理 b=16** | **16** | **22.60** | **77.5ms** | **81.5ms** | **732.2ms** | **58.9%** |

**结论**：同并发窗口下吞吐提升 **4.88×**（4.63 → 22.60 QPS），
TTFT P99 从 **3362ms 降至 81.5ms**，GPU 利用率从 **5.8% 提升到 58.9%**。
串行实现的吞吐与并发**无关**（恒定 ≈4.7 QPS），增加并发只会让延迟线性恶化。

### 批大小扫描曲线（并发固定 16）

| 批大小上限 | 实际峰值批 | QPS | TTFT P50 |
|---|---|---|---|
| 1 | 1 | 2.39 | 6359.1ms |
| 4 | 4 | 5.74 | 2163.9ms |
| 8 | 8 | 11.38 | 777.4ms |
| **16** | **16** | **22.18** | **75.1ms** |
| 32 | 16 | 21.02 | 100.2ms |

吞吐随批大小单调上升至饱和（批大小 = 并发数），继续增大因收益递减而略降 ——
说明**批大小存在最优值**，生产需按 SLA 与显存实测调参。

### 网关纯开销基线（Echo 后端，与并发无关）

| 并发 | QPS | TTFT P50 | TTFT P99 |
|---|---|---|---|
| 16 | 999.79 | 6.0ms | 10.8ms |
| 64 | **1602.59** | 9.8ms | **10.5ms** |

网关自身在 64 并发下可达 **1603 QPS、TTFT P99 仅 10.5ms**，
证明**瓶颈在推理侧而非协议/调度层**，网关可通过增加副本横向扩展。

完整方法与调参建议见 [`docs/性能测试报告.md`](docs/性能测试报告.md)、原始数据见 [`bench/results/`](bench/results/)。

### 复现命令

```bash
python bench/run_bench.py --mode sim --concurrency 1,4,8,16,32 --requests 64 --max-tokens 32 --prompt-tokens 256
python bench/run_bench.py --mode sweep --batch-sweep 1,2,4,8,16,32 --sweep-concurrency 16
python bench/run_bench.py --mode echo --concurrency 1,16,64 --requests 96
python bench/verify_model.py          # 仿真模型自校验（解析预测 vs 实测）
python bench/verify_claims.py         # 文档数字 vs 原始数据对账
```

---

## 测试与质量

| 项目 | 状态 |
|---|---|
| 单元/集成测试 | **263 个用例全部通过**（连跑 4 次验证无 flaky） |
| 覆盖率 | **86%**（唯一未覆盖模块为 `operator/handlers.py`，需真实集群） |
| 静态检查 | ruff（含 ASYNC/SIM/B/RUF 规则集）+ mypy |
| 清单合法性 | 生成的 K8s 清单用官方模型反序列化校验（`V1Deployment`/`V1Service`/`V1HPA` 等） |

```bash
make test        # 全量测试
make cov         # 覆盖率报告
make lint        # ruff + mypy
```

---

## 文档索引

| 文档 | 内容 | 适合谁看 |
|---|---|---|
| [`docs/项目总体设计.md`](docs/项目总体设计.md) | 项目定位、架构全景、技术选型理由、实施路线 | 快速了解全貌 |
| [`docs/架构设计.md`](docs/架构设计.md) | **设计决策与权衡**（为什么这么设计、代价是什么、踩过的坑） | 深入理解设计 |
| [`docs/性能测试报告.md`](docs/性能测试报告.md) | 测试方法、完整数据、调参建议、已知局限 | 关心性能与数据可信度 |
| [`docs/部署与本地验证.md`](docs/部署与本地验证.md) | 从零到跑通：WSL2 → kind → Helm → Operator 验证 → 监控 | 想实际部署运行 |

## 架构

详见 [`docs/项目总体设计.md`](docs/项目总体设计.md) 与 [`docs/架构设计.md`](docs/架构设计.md)。

```
Client ──HTTP/SSE──▶ Ingress ──▶ Gateway(FastAPI)
                                   │  鉴权 / 限流 / 路由
                                   ▼
                          BatchingEngine (连续批处理)
                          优先级队列 → 批构建 → 显存预算
                                   ▼
                     Backend Plugin: vLLM | Transformers | Echo
                                   ▲
        Kubernetes Operator ───────┘   CRD InferenceService
        ├── Reconcile: Deployment / Service / HPA / ServiceMonitor
        ├── Canary: 权重分流 + 错误率自动回滚
        └── Scheduler Plugin(Go): 显存 Filter/Score 装箱
```

---

## 目录结构

```
src/aetherserve/
├── gateway/          # OpenAI 兼容 API 层：路由、鉴权、限流、SSE
├── engine/           # 连续批处理调度引擎（核心）
├── backends/         # 推理后端插件：vllm / transformers / echo
├── observability/    # Prometheus 指标、OTel 追踪、结构化日志
└── operator/         # Kubernetes Operator（CRD + 调和循环 + 灰度）
deploy/               # Helm Chart、K8s 清单、kind 集群定义
bench/                # locust 压测脚本与结果
tests/                # 单元测试与 kind 端到端测试
docs/                 # 项目总体设计、架构设计、性能测试报告、部署指南
```

---

## 开发命令

```bash
make install     # 安装依赖
make dev         # 本地热重载启动网关
make test        # 单元测试
make lint        # ruff + mypy 静态检查
make cov         # 覆盖率报告
make bench       # locust 压测并生成报告
make cluster     # 创建本地 kind 集群
make deploy      # Helm 部署到集群
make e2e         # kind 上的端到端集成测试
```

---

## 设计要点（常见疑问）

<details>
<summary><b>1. 为什么要自研批处理引擎，而不是直接用 vLLM 的 server？</b></summary>

vLLM 的 `--serve` 自带连续批处理，但它是**单进程单模型**的，缺少我们业务需要的三样东西：
**多租户配额与公平性**、**跨模型路由**、**与 K8s 弹性伸缩联动的队列水位信号**。
因此我们把 vLLM 当成"执行器"（无状态、可横向扩展），把准入控制、公平调度、
批构建策略、背压与指标放在网关侧的 BatchingEngine 里。
这样也带来了额外好处：批处理策略可以独立演进和压测，不需要重新编译推理引擎。
</details>

<details>
<summary><b>2. 你的实现是"连续批处理"还是"动态批处理"？（重要边界，请如实说明）</b></summary>

**准确说法：这是"动态批处理（dynamic batching）+ 等满发车"，不是教科书意义上
iteration-level 的连续批处理。** 具体差别：

| 维度 | 本实现 | 真正的连续批处理（如 vLLM） |
|---|---|---|
| 批的组装时机 | 发车前一次性组装，**批派发后不变** | **每个解码步**重新组装 |
| 请求中途进出 | 不支持（批是静态的） | 已完成请求让出、新请求中途补位 |
| 长请求对短请求的影响 | 同批内的短请求仍要等批跑完 | 短请求完成后立即返回 |

**为什么这样取舍**：本项目的调度层定位是"**准入、公平、背压、可观测**"，
真正的 iteration-level 调度交给下游 vLLM 执行器（它内部就是连续批处理）。
这样做的收益是调度策略可以独立演进与压测，不必重新编译推理引擎；
代价是当**网关直连非连续批后端**时，会损失一部分尾延迟优化。

**这个边界的实际影响**：在本项目的仿真与压测里，
批大小被稳定填满（每个解码步 16 个请求），因此吞吐结论不受影响；
但在"批内请求长度差异极大"的 workloads 下，
真正的连续批处理会有更好的尾延迟表现。

**演进方向**：让批执行器支持"某请求完成后从批中移除、从队列补入新请求"，
即把 `_run_batch` 从"跑完一批"改成"维护一个活跃请求集合 + 每步重算"。
这是本项目的 P0 后续任务，详见 [`docs/架构设计.md`](docs/架构设计.md) 的"已知设计局限"一节。
</details>

<details>
<summary><b>3. 显存超卖怎么防？</b></summary>

三层防护：
① **准入控制**：队列中所有在途请求的预估 KV Cache 需求之和不得超过 `gpu_memory_utilization` 预算，
超出直接返回 429（背压），而不是让请求堆积到 OOM；
② **调度器预留**：Go 调度器在 Bind 阶段把显存写入 Pod annotation 作为"预留凭证"，防止多 Pod 争抢同一张卡；
③ **运行时水位**：Prometheus 采集 KV Cache 水位，触发 HPA 扩容或降级到小模型。
</details>

<details>
<summary><b>4. Operator 的调和循环如何做到幂等？</b></summary>

调和函数只做"**读取期望状态 → 读取实际状态 → 计算差异 → 收敛差异**"，不保存任何本地状态。
所有资源通过 `ownerReferences` 关联到 CR，天然获得级联回收；
通过 `status.observedGeneration` 与 `metadata.generation` 对比判断是否需要重新调和。
任何一次调和失败都会以指数退避重试，因此重复执行不产生副作用。
</details>

<details>
<summary><b>5. 灰度发布怎么自动回滚？</b></summary>

发布时创建 canary 版本 Deployment，Ingress 按 `权重 / Header / Cookie` 三种策略分流。
Operator 持续查询 Prometheus 的 `错误率` 与 `P99 延迟`，与基线版本做显著性比较；
连续 N 个采样窗口劣化超过阈值即判定回归，自动把权重归零并回滚，
同时在 `.status.conditions` 中记录 `CanaryFailed` 事件供审计。
</details>

更完整的设计取舍与失败案例见 [`docs/架构设计.md`](docs/架构设计.md)。

---

## Roadmap

- [x] M1 网关层：OpenAI 兼容协议 + SSE 流式 + 多租户鉴权限流
- [x] M2 批处理引擎：优先级队列 + 长度分桶 + 显存预算
- [x] M3 后端插件：Echo / Transformers / vLLM
- [x] M4 可观测性：Prometheus 指标 + Grafana 大盘
- [x] M5 Operator：CRD + 调和循环 + 状态回写
- [x] M6 弹性伸缩：自定义指标 HPA
- [x] M7 GPU 调度器扩展（Go）
- [x] M8 灰度发布与自动回滚
- [ ] M9 模型热加载与多模型共存（进行中）
- [ ] M10 Prefix Cache 复用（规划中）

---

## License

Apache-2.0
