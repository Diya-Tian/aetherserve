# 运维与校验脚本

本目录存放**不参与运行时**、但用于保证项目质量的辅助脚本。
它们都被 `make verify-all` 与 CI 覆盖，是"文档与代码不脱节"的保障机制。

| 脚本 | 用途 | 何时运行 |
|---|---|---|
| [`check_docs.py`](check_docs.py) | 文档一致性：README 相对链接、make 目标、脚本引用、孤儿文档、隐私检查 | 改文档或 Makefile 后，CI 必跑 |
| [`check_readme.py`](check_readme.py) | README 声明 vs 仓库现实：声明的路径是否真实存在、是否声称了未实现的组件、是否含个人求职材料 | 改 README 后，CI 必跑 |

> 其他校验脚本按职责就近放置，便于与所校验的对象一起维护：
> - `bench/verify_model.py` — 仿真模型自校验（解析预测 vs 实测）
> - `bench/verify_claims.py` — 文档数字 vs 压测原始数据对账
> - `deploy/helm/verify_chart.py` — Helm 渲染结果的官方模型反序列化校验
> - `deploy/k8s/generate_manifests.py` — 从代码生成 CRD 清单

---

## 为什么需要这些脚本？

项目里有四类"**容易悄悄腐化**"的一致性：

| 腐化类型 | 后果 | 对应脚本 |
|---|---|---|
| 代码改了但 YAML 没重新生成 | CRD 与实现不符，用户写出跑不起来的 CR | `generate_manifests.py` + CI diff 检查 |
| 文档里的数字与实测不一致 | 评审或排查时被追问穿帮 | `verify_claims.py` |
| 仿真实现有 bug 但结果"看起来合理" | 结论不可信 | `verify_model.py` |
| README 链接失效 / make 目标不存在 / 孤儿文档 | 项目显得不可信、不可用 | `check_docs.py` |
| **README 声称了不存在的东西**（目录、组件、文档） | 读者按图索骥找不到，直接怀疑项目真实性 | `check_readme.py` |
| **个人求职材料混入开源仓库** | 泄露手机号等隐私，且与项目工程无关 | `check_docs.py` / `check_readme.py` 的隐私检查 |

把一致性检查写成脚本而不是"靠人记得"，是这套工程质量的基础。

---

## 使用方式

```bash
# 全部校验（建议提交前跑）
make verify-all

# 单独运行
make verify          # 仿真模型自校验
make verify-claims   # 文档数字对账
make verify-chart    # Helm Chart 校验（需要 helm 在 PATH）
make verify-docs     # 文档一致性

# 直接调用
python tools/check_docs.py
```

**依赖说明**：`verify-chart` 需要 `helm` CLI。
Windows 安装：`winget install Helm.Helm`（安装后需重开 shell 让 PATH 生效）。
CI 中使用 `azure/setup-helm@v4` 自动准备。
