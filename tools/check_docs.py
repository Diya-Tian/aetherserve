"""文档一致性检查：README 引用的文件/命令是否真实存在。

为什么需要它？
-------------
文档里的链接和命令如果指向不存在的文件，会直接损害项目可信度
（"README 说的东西根本跑不起来"）。这个小脚本把这类问题自动化，
纳入 `make verify-all` 一起跑。
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def tracked_files() -> list[str]:
    """列出 git 已跟踪的文件。

    用 ``git ls-files`` 而不是遍历目录，因为只有它反映"真正会进仓库的内容" ——
    遍历目录会把未跟踪的临时文件也算进来，产生误报。
    """
    try:
        result = subprocess.run(
            ["git", "ls-files"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if result.returncode == 0:
            return [line.strip().replace("\\", "/") for line in result.stdout.splitlines() if line.strip()]
    except FileNotFoundError:
        pass
    excluded = {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache", "acl-recovery"}
    return [
        str(p.relative_to(REPO_ROOT)).replace("\\", "/")
        for p in REPO_ROOT.rglob("*")
        if p.is_file() and not any(part in excluded for part in p.parts)
    ]


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
    # 扫描范围：仓库内所有 Markdown（含 tools/、deploy/ 等子目录）
    scan_files = [REPO_ROOT / "README.md", *docs]
    scan_files += sorted(REPO_ROOT.glob("tools/*.md"))
    scan_files += sorted(REPO_ROOT.glob("deploy/**/*.md"))
    all_md = "\n".join(p.read_text(encoding="utf-8") for p in scan_files if p.exists())
    for doc in docs:
        name = doc.name
        # 统计在 README + 其他文档中被引用的次数（自身引用不计）
        mentions = all_md.count(name)
        if mentions > 1:
            print(f"  [OK  ] {name}（被引用 {mentions - 1} 次）")
        else:
            ok = False
            print(f"  [WARN] {name}（未被任何文档引用）")

    # 5) 隐私保护：开源仓库中不得出现个人求职材料
    #    简历与求职说明属于个人材料，与项目工程无关，且常含手机号等隐私信息。
    print("\n[5] 隐私检查：仓库内不得含个人求职材料")
    files = tracked_files()
    personal = ("简历", "resume/", "cover-letter", "求职")
    offenders = [f for f in files if any(pat in f for pat in personal)]
    if offenders:
        ok = False
        print("  [FAIL] 发现个人求职材料（必须移出仓库）:")
        for f in offenders:
            print(f"         - {f}")
    else:
        print(f"  [OK  ] 已检查 {len(files)} 个受版本控制的文件，无个人求职材料")

    print()
    print("=" * 78)
    print("结论:", "全部一致 ✅" if ok else "存在不一致 ❌")
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
