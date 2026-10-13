# 贡献指南

感谢关注 AetherServe。本文件说明本项目的开发约定，目标是让改动**可预测、可验证、可回溯**。

---

## 1. 环境准备

```bash
git clone https://github.com/<your-name>/aetherserve.git
cd aetherserve
make install       # 创建 .venv 并安装 [dev,operator] 依赖
make test          # 确认基线全绿（应为 245 passed）
```

> 无需 GPU 即可开发：默认 Echo 后端是确定性假后端，跑通协议与调度链路；
> 性能相关的仿真后端也不需要 GPU（见 `src/aetherserve/sim/`）。

---

## 2. 分支与提交

- 分支命名：`feat/短描述`、`fix/短描述`、`docs/短描述`、`perf/短描述`
- 提交信息遵循 [Conventional Commits](https://www.conventionalcommits.org/)：
  - `feat(engine): 支持按 prompt 长度分桶减少 padding`
  - `fix(queue): 修复老化重排未生效导致低优先级请求饿死`
  - `perf(sim): 批同步改为屏障模型，消除重复计费`
  - `docs(bench): 补充批大小扫描曲线与调参建议`

**提交信息里请写清"为什么"**，而不只是"改了什么"。示例：

```
fix(engine): 队列关闭后分发循环空转，饿死事件循环

stop() 先停分发循环再关队列时，get() 会因队列关闭立即返回空列表，
导致 while 循环变成忙等，已提交请求的消费者协程永远拿不到执行机会。
改为"先置位、再排空、最后关队列"，并在空结果时 await sleep(0) 让出。
```

---

## 3. 代码约定

### 3.1 分层依赖必须单向

```
backends → gateway.schemas        （只依赖协议模型，不依赖 app）
gateway.app → engine → backends    （装配方向）
operator → crd/builders/canary → reconciler → handlers
```

新增模块前请先确认依赖方向，**不要引入反向依赖**（历史上 `gateway/__init__`
抢先导入 `app` 造成过循环导入，现用 PEP 562 惰性导出解决）。

### 3.2 核心逻辑写成纯函数

调度、批构建、灰度决策、清单生成都实现为**无 I/O 的纯函数**。这样做的好处是
可以用毫秒级单测覆盖全部边界，而不必"改集群等 5 分钟"。

新增功能时请沿用这个模式：**纯逻辑与 I/O 分离**。

### 3.3 异步代码

- 禁止在协程中使用阻塞调用（`time.sleep`、同步 HTTP、文件大 IO）；
- 必要的阻塞操作请用 `asyncio.to_thread` 包装；
- 循环里若可能出现"立即返回"的分支，务必 `await asyncio.sleep(0)` 让出事件循环，
  否则会忙等饿死其他任务（本项目踩过两次）。

### 3.4 类型与风格

```bash
make lint        # ruff check（含 ASYNC / SIM / B / RUF 规则集）
make fmt         # ruff 自动修复 + 格式化
make typecheck   # mypy
```

- 行宽 110，目标 Python 3.11+；
- 公共函数必须有类型注解；
- 注释解释**设计理由与权衡**，不要复述代码字面含义。

---

## 4. 测试要求

| 改动类型 | 必须补充的测试 |
|---|---|
| 新增/修改调度策略 | 不变量测试（批大小上界、显存记账归零、优先级/老化单调性） |
| 新增 K8s 资源类型 | 清单能被官方模型反序列化 + selector 与 pod 标签一致性 |
| 修改灰度逻辑 | 边界用例（样本不足、错误率恰好等于阈值、终态不再推进） |
| 修改性能模型 | `bench/verify_model.py` 必须 PASS（偏差 < 15%） |

**禁止写依赖墙钟的脆弱断言**：Windows 定时器粒度约 15ms，
毫秒级阈值断言会间歇性失败。请断言**计费模型/计数值**等确定性量，
把带时间的验证留给 `bench/` 下的压测脚本。

```bash
make test    # 全量
make cov     # 覆盖率（当前 86%）
make verify  # 仿真模型自校验
```

> 提交前请**连跑 3 次** `make test`，确认没有 flaky 用例。

---

## 5. 性能相关改动

任何声称"提升性能"的改动都必须附带可复现数据：

```bash
git checkout main && make bench > /tmp/before.md     # 基线
git checkout feat/xxx && make bench > /tmp/after.md  # 改动后
```

PR 描述中请贴出对比表（QPS / TTFT P99 / E2E P99），并说明测试条件
（并发、prompt 长度、生成长度、批大小配置）。

⚠️ **不要把未实测的数字写进文档或简历**。当前性能报告中的数据来自
耗时模型仿真，报告中已明确标注该边界；若你补齐了真实 GPU 数据，请同时更新
`docs/性能测试报告.md` 与 `README.md` 的声明。

---

## 6. CRD 变更流程

CRD 是**对外契约**，变更需谨慎：

1. 修改 `src/aetherserve/operator/crd.py` 中的 dataclass 与 `crd_manifest()`；
2. 运行 `python deploy/k8s/generate_manifests.py --all --out deploy/k8s/aetherserve-crd.yaml`；
3. 检查生成的 YAML diff 是否符合预期；
4. 更新 `tests/test_operator_builders.py` 中的 schema 断言；
5. **破坏性变更**需新增 API 版本（如 `v1alpha2`）而非直接修改 `v1alpha1`，
   以兼容已存在的 CR。

---

## 7. 提交 PR

- PR 标题同提交信息规范；
- 描述中请包含：**问题背景 / 方案与权衡 / 验证方式 / 影响面**；
- 确保 CI 全绿（lint + test + benchmark + image build）；
- 涉及架构决策的改动，请同步更新 `docs/项目总体设计.md` 或 `docs/架构设计.md`。

---

## 8. 行为准则

- 讨论对事不对人，技术分歧用**数据与复现步骤**说话；
- 欢迎提出反对意见，尤其是能指出设计缺陷或虚假性能主张的意见。
