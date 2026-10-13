"""校验 Helm 渲染结果：结构 + 官方模型反序列化 + RBAC 权限自检。

为什么需要这个脚本？
-------------------
`helm lint` 只能检查模板语法，无法发现"渲染出的 YAML 结构合法但不是合法 K8s 对象"
这类问题（比如字段名拼错、类型不对）。
本脚本用 Kubernetes 官方客户端的反序列化器做类型化校验，
并对渲染出的 Deployment 断言"selector 与 pod 标签一致"等关键不变量。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CHART = REPO_ROOT / "deploy" / "helm" / "aetherserve"

# Kind → Kubernetes 官方模型名（用于反序列化校验）
MODEL_MAP = {
    "Deployment": "V1Deployment",
    "Service": "V1Service",
    "ServiceAccount": "V1ServiceAccount",
    "ClusterRole": "V1ClusterRole",
    "ClusterRoleBinding": "V1ClusterRoleBinding",
    "CustomResourceDefinition": "V1CustomResourceDefinition",
}


def render() -> list[dict[str, Any]]:
    """渲染 Chart 并解析为对象列表。

    注意：Windows 上 subprocess 默认按系统编码（GBK）解码子进程输出，
    而 helm 输出是 UTF-8（我们的模板里有中文注释），
    因此必须显式指定 encoding 并容错，否则会抛 UnicodeDecodeError。
    """
    result = subprocess.run(
        ["helm", "template", "aetherserve", str(CHART), "--namespace", "ai-serving"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        print("!! helm template 失败：")
        print(result.stderr)
        print("\n提示：需要 helm 在 PATH 中。Windows 可用 winget install Helm.Helm 安装，")
        print("      或把 helm.exe 所在目录加入 PATH 后重试。")
        sys.exit(2)
    return [d for d in yaml.safe_load_all(result.stdout) if d]


def deserialize(manifest: dict[str, Any], model: str) -> Any:
    from kubernetes import client

    return client.ApiClient()._ApiClient__deserialize(manifest, model)


def main() -> int:
    docs = render()
    print("=" * 78)
    print(f"Helm 渲染结果校验（共 {len(docs)} 个对象）")
    print("=" * 78)

    ok = True
    kinds: list[str] = []
    for doc in docs:
        kind = doc.get("kind", "")
        name = doc.get("metadata", {}).get("name", "?")
        kinds.append(kind)

        model = MODEL_MAP.get(kind)
        if model is None:
            print(f"  [SKIP] {kind}/{name}（无对应官方模型，可能是 CRD 类型）")
            continue
        try:
            deserialize(doc, model)
            print(f"  [OK  ] {kind}/{name}")
        except Exception as exc:
            ok = False
            print(f"  [FAIL] {kind}/{name}: {type(exc).__name__}: {exc}")

    print()
    print("-" * 78)
    print("关键不变量检查")
    print("-" * 78)

    # 1) 必须包含 CRD / RBAC / Operator Deployment
    required = {
        "CustomResourceDefinition",
        "ClusterRole",
        "ClusterRoleBinding",
        "Deployment",
        "ServiceAccount",
    }
    missing = required - set(kinds)
    if missing:
        ok = False
        print(f"  [FAIL] 缺少必需对象类型: {missing}")
    else:
        print(f"  [OK  ] 必需对象齐备: {sorted(required)}")

    # 2) Deployment 的 selector 必须能匹配自己的 Pod 模板标签
    for doc in docs:
        if doc.get("kind") != "Deployment":
            continue
        selector = doc["spec"]["selector"]["matchLabels"]
        pod_labels = doc["spec"]["template"]["metadata"]["labels"]
        mismatched = {k: (v, pod_labels.get(k)) for k, v in selector.items() if pod_labels.get(k) != v}
        if mismatched:
            ok = False
            print(f"  [FAIL] Deployment/{doc['metadata']['name']} selector 与 pod 标签不一致: {mismatched}")
        else:
            print(f"  [OK  ] Deployment/{doc['metadata']['name']} selector ⊇ pod 标签")

    # 3) Service 的 selector 必须能选中某个 Deployment 的 Pod
    pod_label_sets = [
        d["spec"]["template"]["metadata"]["labels"] for d in docs if d.get("kind") == "Deployment"
    ]
    for doc in docs:
        if doc.get("kind") != "Service":
            continue
        sel = doc["spec"]["selector"]
        hit = any(all(pod.get(k) == v for k, v in sel.items()) for pod in pod_label_sets)
        if hit:
            print(f"  [OK  ] Service/{doc['metadata']['name']} 能选中后端 Pod")
        else:
            ok = False
            print(f"  [FAIL] Service/{doc['metadata']['name']} 选不到任何 Pod！selector={sel}")

    # 4) ClusterRole 不得使用通配符（最小权限原则）
    for doc in docs:
        if doc.get("kind") != "ClusterRole":
            continue
        wildcards = [
            r for r in doc.get("rules", []) if "*" in r.get("verbs", []) or "*" in r.get("resources", [])
        ]
        if wildcards:
            ok = False
            print(f"  [FAIL] ClusterRole/{doc['metadata']['name']} 使用了通配符权限（违反最小权限原则）")
        else:
            total = sum(len(r.get("verbs", [])) for r in doc.get("rules", []))
            print(
                f"  [OK  ] ClusterRole/{doc['metadata']['name']} 无通配符，共 {len(doc.get('rules', []))} 条规则 / {total} 个动词"
            )

    # 5) CRD 必须带 keep 注解（防止 helm uninstall 级联删除所有 CR）
    for doc in docs:
        if doc.get("kind") != "CustomResourceDefinition":
            continue
        ann = doc["metadata"].get("annotations", {}) or {}
        if ann.get("helm.sh/resource-policy") == "keep":
            print("  [OK  ] CRD 带 helm.sh/resource-policy=keep（防止误删 CR）")
        else:
            ok = False
            print("  [FAIL] CRD 缺少 keep 注解：helm uninstall 会级联删除所有 InferenceService！")

    # 6) CRD 的 backend 枚举必须与代码中的后端注册表一致
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from aetherserve.backends import available_backends

    for doc in docs:
        if doc.get("kind") != "CustomResourceDefinition":
            continue
        schema = doc["spec"]["versions"][0]["schema"]["openAPIV3Schema"]
        declared = set(schema["properties"]["spec"]["properties"]["backend"]["enum"])
        expected = set(available_backends())
        if declared == expected:
            print(f"  [OK  ] CRD backend 枚举与后端注册表一致: {sorted(declared)}")
        else:
            ok = False
            print(f"  [FAIL] CRD backend 枚举 {sorted(declared)} ≠ 注册表 {sorted(expected)}")

    # 7) 安全上下文：不允许特权、不允许提权
    for doc in docs:
        if doc.get("kind") != "Deployment":
            continue
        for c in doc["spec"]["template"]["spec"]["containers"]:
            sc = c.get("securityContext", {})
            bad = []
            if sc.get("privileged"):
                bad.append("privileged=true")
            if sc.get("allowPrivilegeEscalation") is not False:
                bad.append("allowPrivilegeEscalation 未显式设为 false")
            if bad:
                ok = False
                print(f"  [FAIL] Deployment/{doc['metadata']['name']} 容器安全上下文有问题: {bad}")
            else:
                print(f"  [OK  ] Deployment/{doc['metadata']['name']} 容器已做安全加固")

    print()
    print("=" * 78)
    print("结论:", "全部通过 ✅" if ok else "存在问题 ❌")
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
