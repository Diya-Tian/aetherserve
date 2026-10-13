"""文档一致性检查：README 引用的文件/命令是否真实存在。

为什么需要它？
-------------
文档里的链接和命令如果指向不存在的文件，会直接损害项目可信度
（"README 说的东西根本跑不起来"）。这个小脚本把这类问题自动化，
纳入 `make verify-all` 一起跑。
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ok = True
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    print("=" * 78)
    print("文档一致性检查")
    print("=" * 78)

    # 1) README 中引用的相对文件路径是否都存在
    print("\n[1] README 引用的文件")
    refs = sorted(set(re.findall(r"\]\(([^)#][^)]*)\)", readme)))
    checked = 0
    for ref in refs:
        if ref.startswith(("http://", "https://", "mailto:")):
            continue
        path = REPO_ROOT / ref
        checked += 1
        if path.exists():
            print(f"  [OK  ] {ref}")
        else:
            ok = False
            print(f"  [MISS] {ref}")
    print(f"  → 检查了 {checked} 个相对路径引用")

    # 2) README 中提到的 make 目标是否都在 Makefile 中定义
    print("\n[2] README 里提到的 make 目标")
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    defined = set(re.findall(r"^([a-zA-Z][a-zA-Z0-9_-]*):", makefile, re.MULTILINE))
    mentioned = sorted(set(re.findall(r"^make ([a-zA-Z][a-zA-Z0-9_-]*)", readme, re.MULTILINE)))
    for target in mentioned:
        if target in defined:
            print(f"  [OK  ] make {target}")
        else:
            ok = False
            print(f"  [MISS] make {target}（Makefile 中未定义）")
    print(f"  → README 提到 {len(mentioned)} 个目标，Makefile 共定义 {len(defined)} 个")

    # 3) MAKE 目标里引用的脚本文件是否存在
    print("\n[3] Makefile 目标引用的脚本")
    script_refs = sorted(
        set(re.findall(r"(?:python|pytest)\s+((?:bench|deploy|tests)/[\w./-]+\.py)", makefile))
    )
    for ref in script_refs:
        path = REPO_ROOT / ref
        if path.exists():
            print(f"  [OK  ] {ref}")
        else:
            ok = False
            print(f"  [MISS] {ref}")
    print(f"  → 检查了 {len(script_refs)} 个脚本引用")

    # 4) docs/ 下每个文档都应被 README 或其它文档引用（避免"写了没人看的孤儿文档"）
    print("\n[4] docs/ 文档的可达性")
    docs = sorted((REPO_ROOT / "docs").glob("*.md"))
    all_md = "\n".join(p.read_text(encoding="utf-8") for p in [REPO_ROOT / "README.md", *docs] if p.exists())
    for doc in docs:
        name = doc.name
        # 统计在 README + 其他文档中被引用的次数（自身引用不计）
        mentions = all_md.count(name)
        if mentions > 1:
            print(f"  [OK  ] {name}（被引用 {mentions - 1} 次）")
        else:
            ok = False
            print(f"  [WARN] {name}（未被任何文档引用）")

    print()
    print("=" * 78)
    print("结论:", "全部一致 ✅" if ok else "存在不一致 ❌")
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
