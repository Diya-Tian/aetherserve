"""README 声明一致性检查：文档里写的路径/功能必须在仓库里真实存在。

为什么需要它？
-------------
README 最容易出现的"信任杀手"是**声称了不存在的东西**：
- 目录结构里列了 `scheduler-plugin/`，但那个目录从没实现；
- 架构图里画了某个组件，代码里根本没有；
- 索引里链接了一个已删除的文档。

这类问题比"缺少文档"更糟 —— 读者按图索骥找不到，会直接怀疑整个项目的真实性。
本脚本把"文档声明 vs 仓库现实"的核对自动化。
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
README = REPO_ROOT / "README.md"

# 只检查这些顶层目录/文件，避免把架构图里的示意名称误判为路径
TRACKED_ROOTS = {
    ".github",
    "bench",
    "deploy",
    "docs",
    "src",
    "tests",
    "tools",
    "resume",
    "scheduler-plugin",
    "CONTRIBUTING.md",
    "Dockerfile",
    "LICENSE",
    "Makefile",
    "README.md",
    "pyproject.toml",
}


def tracked_files() -> set[str]:
    result = subprocess.run(
        ["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, encoding="utf-8", check=False
    )
    return {line.strip().replace("\\", "/") for line in result.stdout.splitlines() if line.strip()}


def extract_code_blocks(text: str) -> list[str]:
    return re.findall(r"```[a-zA-Z]*\n(.*?)```", text, re.S)


def main() -> int:
    ok = True
    text = README.read_text(encoding="utf-8")
    tracked_full = tracked_files()

    print("=" * 78)
    print("README 声明一致性检查")
    print("=" * 78)

    # ---- 1) 代码块里出现的顶层路径必须真实存在
    print("\n[1] 代码块中声明的路径")
    claimed: set[str] = set()
    for block in extract_code_blocks(text):
        for line in block.splitlines():
            # 跳过 shell 命令块（以 $ / curl / make / python 等开头）
            stripped = line.strip()
            if not stripped or stripped.startswith(
                (
                    "$",
                    "#",
                    "curl",
                    "make",
                    "python",
                    "pip",
                    "docker",
                    "kubectl",
                    "helm",
                    "git",
                    "cd",
                    "export",
                    "wsl",
                    "winget",
                    "for ",
                    "if ",
                    "echo",
                    "set ",
                )
            ):
                continue
            # 抓取行首形如 "src/aetherserve/" 或 "scheduler-plugin/" 的路径
            m = re.match(r"^[│├└─\s]*([A-Za-z0-9_.-]+/[A-Za-z0-9_./-]*)", stripped)
            if m:
                claimed.add(m.group(1).rstrip("/"))

    missing_top: list[str] = []
    for c in sorted(claimed):
        top = c.split("/")[0]
        if top not in TRACKED_ROOTS:
            continue
        # 注意：git ls-files 只列文件、不列目录，
        # 因此目录型路径要用文件系统判断（docs/ 这类目录真实存在但没有对应条目）。
        exists = (REPO_ROOT / c).exists() or any(f.startswith(c + "/") for f in tracked_full)
        if not exists:
            missing_top.append(c)
            ok = False

    if missing_top:
        print("  [FAIL] 代码块中声明但仓库不存在的路径:")
        for m in sorted(set(missing_top)):
            print(f"         - {m}")
    else:
        print(f"  [OK  ] 检查了 {len(claimed)} 个声明路径，均真实存在")

    # ---- 2) README 相对链接必须存在
    print("\n[2] README 相对链接")
    links = sorted(set(re.findall(r"\]\((?!https?://|mailto:|#)([^)#]+)\)", text)))
    dead = [ln for ln in links if not (REPO_ROOT / ln).exists()]
    if dead:
        ok = False
        print("  [FAIL] 失效链接:")
        for d in dead:
            print(f"         - {d}")
    else:
        print(f"  [OK  ] 检查了 {len(links)} 个相对链接，全部有效")

    # ---- 3) 不得声称未实现的功能目录
    print("\n[3] 功能声称与代码的对应")
    # 这些目录若在文档里被描述为"已实现"，必须真实存在
    unimplemented_markers = ["scheduler-plugin", "resume/"]
    bad_claims: list[str] = []
    for marker in unimplemented_markers:
        if marker in text and marker.split("/")[0] not in {"resume"}:
            # 出现在代码块或正文即视为声称
            for i, line in enumerate(text.splitlines(), 1):
                if marker in line:
                    bad_claims.append(f"L{i}: {line.strip()[:70]}")
    if bad_claims:
        ok = False
        print("  [FAIL] README 提到未实现的组件:")
        for b in bad_claims:
            print(f"         {b}")
    else:
        print("  [OK  ] 未声称任何不存在的组件")

    # ---- 4) 文档索引里列出的 docs/ 文件必须存在
    print("\n[4] 文档索引表")
    doc_refs = re.findall(r"\(docs/([^)]+\.md)\)", text)
    missing_doc = [d for d in doc_refs if not (REPO_ROOT / "docs" / d).exists()]
    if missing_doc:
        ok = False
        print("  [FAIL] 索引指向不存在的文档:")
        for d in missing_doc:
            print(f"         - docs/{d}")
    else:
        print(f"  [OK  ] 索引中 {len(doc_refs)} 个文档均存在")

    # ---- 5) 仓库中不得出现个人求职材料
    print("\n[5] 隐私：不得含个人求职材料")
    personal = ("简历", "cover-letter", "求职")
    offenders = sorted(f for f in tracked_full if any(p in f for p in personal))
    if offenders:
        ok = False
        print("  [FAIL] 发现个人求职材料:")
        for o in offenders:
            print(f"         - {o}")
    else:
        print(f"  [OK  ] 已检查 {len(tracked_full)} 个受控文件，无个人求职材料")

    print()
    print("=" * 78)
    print("结论:", "README 与仓库一致 ✅" if ok else "README 存在不实声明 ❌")
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
