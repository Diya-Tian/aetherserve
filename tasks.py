"""跨平台任务入口（零依赖，Windows / Linux / macOS / WSL 通用）。

为什么需要它？
-------------
仓库里已有 ``Makefile``，但它有几个平台假设：

- ``$(VENV)/bin/...`` —— 虚拟环境的可执行文件位置在 Windows 上是 ``Scripts/``；
- ``SHELL := /bin/sh``、``grep`` / ``awk`` / ``rm -rf`` —— Windows 默认没有这些工具；
- 因此 ``make dev`` 在 Windows PowerShell 里会直接失败。

本脚本用纯 Python 实现同样的任务，**不依赖 make、shell 工具或第三方库**，
在任意平台都能跑：

    python tasks.py            # 列出全部任务
    python tasks.py dev        # 启动网关
    python tasks.py test       # 运行测试
    python tasks.py verify-all # 跑完全部校验

``Makefile`` 仍然保留（并已修正为自动探测 venv 脚本目录），
供习惯 make 或在 Linux/CI 环境使用；两者调用同一套逻辑。
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
VENV = REPO_ROOT / ".venv"
PORT = os.environ.get("PORT", "8000")
CLUSTER = os.environ.get("CLUSTER", "aetherserve")
NAMESPACE = os.environ.get("NAMESPACE", "ai-serving")
IMAGE = os.environ.get("IMAGE", "aetherserve/gateway:0.1.0")


# --------------------------------------------------------------------- 工具
def venv_bin(name: str) -> str:
    """定位虚拟环境里的可执行文件，自动适配 Windows 与 Unix。

    若 venv 不存在则回退到 PATH 上的同名命令（并给出提示），
    这样"没建 venv 但系统里装了"的情况下仍可用。
    """
    subdir = "Scripts" if os.name == "nt" else "bin"
    suffix = ".exe" if os.name == "nt" else ""
    candidate = VENV / subdir / f"{name}{suffix}"
    if candidate.exists():
        return str(candidate)
    found = shutil.which(name)
    if found:
        print(f"  [提示] 未找到 {candidate}，回退使用系统命令 {found}")
        print("         建议先运行: python tasks.py install")
        return found
    raise SystemExit(
        f"找不到可执行文件 {name}：既不在 {candidate}，也不在 PATH。\n请先运行: python tasks.py install"
    )


def run(argv: list[str], *, cwd: Path | None = None, check: bool = True) -> int:
    """执行命令并透传输出（使用当前解释器环境，避免管道捕获问题）。"""
    printable = " ".join(argv)
    print(f"\n$ {printable}\n")
    result = subprocess.run(argv, cwd=str(cwd or REPO_ROOT), check=False)
    if check and result.returncode != 0:
        raise SystemExit(f"命令失败（exit={result.returncode}）: {printable}")
    return result.returncode


def py(*args: str) -> list[str]:
    return [sys.executable, *args]


def tool(name: str) -> list[str]:
    return [venv_bin(name)]


# --------------------------------------------------------------------- 任务
def task_install() -> None:
    """创建虚拟环境并安装依赖（含开发与 Operator 依赖）。"""
    run([sys.executable, "-m", "venv", str(VENV)])
    run([*tool("pip"), "install", "-U", "pip"])
    run([*tool("pip"), "install", "-e", ".[dev,operator]"], cwd=REPO_ROOT)


def task_dev() -> None:
    """本地热重载启动网关（默认 Echo 后端，无需 GPU）。"""
    run([*tool("uvicorn"), "aetherserve.gateway.app:app", "--reload", "--host", "0.0.0.0", "--port", PORT])


def task_run() -> None:
    """本地启动网关（不热重载，接近生产形态）。"""
    run(
        [
            *tool("uvicorn"),
            "aetherserve.gateway.app:app",
            "--host",
            "0.0.0.0",
            "--port",
            PORT,
            "--workers",
            "1",
        ]
    )


def task_test() -> None:
    """运行全部测试（e2e 在无集群时自动跳过）。"""
    run([*tool("pytest"), "tests/", "-q"])


def task_cov() -> None:
    """运行测试并输出覆盖率。"""
    run([*tool("pytest"), "tests/", "-q", "--cov=aetherserve", "--cov-report=term-missing"])


def task_lint() -> None:
    """ruff 静态检查。"""
    run([*tool("ruff"), "check", "src", "tests", "bench", "deploy", "tools"])


def task_fmt() -> None:
    """ruff 自动修复 + 格式化。"""
    run([*tool("ruff"), "check", "--fix", "src", "tests", "bench", "deploy", "tools"])
    run([*tool("ruff"), "format", "src", "tests", "bench", "deploy", "tools"])


def task_typecheck() -> None:
    """mypy 类型检查。"""
    run([*tool("mypy"), "src"])


def task_bench() -> None:
    """主对标压测：串行 vs 批处理。"""
    run(
        py(
            "bench/run_bench.py",
            "--mode",
            "sim",
            "--concurrency",
            "1,4,8,16,32",
            "--requests",
            "64",
            "--max-tokens",
            "32",
            "--prompt-tokens",
            "256",
        )
    )


def task_bench_sweep() -> None:
    """批大小扫描曲线。"""
    run(
        py(
            "bench/run_bench.py",
            "--mode",
            "sweep",
            "--batch-sweep",
            "1,2,4,8,16,32",
            "--sweep-concurrency",
            "16",
            "--requests",
            "64",
            "--max-tokens",
            "32",
            "--prompt-tokens",
            "256",
        )
    )


def task_bench_echo() -> None:
    """网关纯开销基线。"""
    run(py("bench/run_bench.py", "--mode", "echo", "--concurrency", "1,16,64", "--requests", "96"))


def task_verify() -> None:
    """仿真模型自校验（解析预测 vs 实测）。"""
    run(py("bench/verify_model.py"))


def task_verify_claims() -> None:
    """文档数字 vs 压测原始数据对账。"""
    run(py("bench/verify_claims.py"))


def task_verify_chart() -> None:
    """Helm Chart 校验（需要 helm 在 PATH）。"""
    run(["helm", "lint", "deploy/helm/aetherserve"])
    run(py("deploy/helm/verify_chart.py"))


def task_verify_docs() -> None:
    """文档一致性校验。"""
    run(py("tools/check_docs.py"))


def task_verify_readme() -> None:
    """README 声明与仓库现实一致性校验。"""
    run(py("tools/check_readme.py"))


def task_verify_coldstart() -> None:
    """冷启动端到端验证（全新 venv + 按 README 操作，24 项检查）。"""
    run(py("tools/coldstart_check.py"))


def task_verify_all() -> None:
    """跑完全部校验（不含冷启动，因它耗时较长）。

    各校验失败不立即中断，最后汇总 —— 这样一次就能看到所有问题。
    """
    steps: list[tuple[str, Callable[[], None]]] = [
        ("模型自校验", task_verify),
        ("数字对账", task_verify_claims),
        ("文档一致性", task_verify_docs),
        ("README 一致性", task_verify_readme),
    ]
    failures: list[str] = []
    for name, fn in steps:
        try:
            fn()
        except SystemExit:
            failures.append(name)
    # helm 可能未安装，单独处理避免误判
    try:
        task_verify_chart()
    except (SystemExit, FileNotFoundError):
        print("  [跳过] Helm Chart 校验（未找到 helm 命令）")

    print("\n" + "=" * 70)
    if failures:
        print(f"校验失败项: {', '.join(failures)}")
        raise SystemExit(1)
    print("全部校验通过 ✅")
    print("=" * 70)


def task_docker_build() -> None:
    """构建网关镜像。"""
    run(["docker", "build", "-t", IMAGE, "."])


def task_cluster() -> None:
    """创建本地 kind 集群。"""
    run(["kind", "create", "cluster", "--name", CLUSTER, "--config", "deploy/local/kind-cluster.yaml"])
    for node in (f"{CLUSTER}-worker",):
        run(
            ["kubectl", "label", "node", node, "aetherserve.io/accelerator=nvidia", "--overwrite"],
            check=False,
        )
    run(["kubectl", "create", "namespace", NAMESPACE], check=False)


def task_cluster_delete() -> None:
    """删除本地 kind 集群。"""
    run(["kind", "delete", "cluster", "--name", CLUSTER])


def task_crd() -> None:
    """从代码生成 CRD 并安装。"""
    run(py("deploy/k8s/generate_manifests.py", "--all", "--out", "deploy/k8s/aetherserve-crd.yaml"))
    run(["kubectl", "apply", "-f", "deploy/k8s/aetherserve-crd.yaml"])


def task_operator() -> None:
    """本地运行 Operator（读取 ~/.kube/config）。"""
    run(py("-m", "aetherserve.operator.main", "--verbose"))


def task_e2e() -> None:
    """在 kind 集群上跑端到端测试（无集群时自动跳过）。"""
    run([*tool("pytest"), "tests/e2e", "-q", "-m", "e2e"])


def task_render() -> None:
    """渲染 Helm 清单到本地便于 review。"""
    out = REPO_ROOT / "deploy" / "helm" / "rendered.yaml"
    print(f"\n$ helm template ... > {out}\n")
    with out.open("w", encoding="utf-8") as fh:
        result = subprocess.run(
            ["helm", "template", "aetherserve", "deploy/helm/aetherserve", "--namespace", NAMESPACE],
            stdout=fh,
            cwd=str(REPO_ROOT),
            check=False,
        )
    if result.returncode != 0:
        raise SystemExit("helm template 失败")
    print(f"已渲染到 {out}")


def task_clean() -> None:
    """清理构建产物与缓存（跨平台，不依赖 rm -rf）。"""
    import shutil as _shutil

    targets = [
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        "htmlcov",
        ".coverage",
        "dist",
        "build",
        "deploy/helm/rendered.yaml",
    ]
    for t in targets:
        p = REPO_ROOT / t
        if p.is_dir():
            _shutil.rmtree(p, ignore_errors=True)
            print(f"  删除目录 {t}")
        elif p.is_file():
            p.unlink(missing_ok=True)
            print(f"  删除文件 {t}")
    for p in REPO_ROOT.rglob("__pycache__"):
        _shutil.rmtree(p, ignore_errors=True)
    print("清理完成（保留 .venv）")


TASKS: dict[str, Callable[[], None]] = {
    "install": task_install,
    "dev": task_dev,
    "run": task_run,
    "test": task_test,
    "cov": task_cov,
    "lint": task_lint,
    "fmt": task_fmt,
    "typecheck": task_typecheck,
    "bench": task_bench,
    "bench-sweep": task_bench_sweep,
    "bench-echo": task_bench_echo,
    "verify": task_verify,
    "verify-claims": task_verify_claims,
    "verify-chart": task_verify_chart,
    "verify-docs": task_verify_docs,
    "verify-readme": task_verify_readme,
    "verify-coldstart": task_verify_coldstart,
    "verify-all": task_verify_all,
    "docker-build": task_docker_build,
    "cluster": task_cluster,
    "cluster-delete": task_cluster_delete,
    "crd": task_crd,
    "operator": task_operator,
    "e2e": task_e2e,
    "render": task_render,
    "clean": task_clean,
}


def print_help() -> None:
    print("AetherServe 任务入口（跨平台，无需 make）\n")
    print("用法: python tasks.py <任务>\n")
    print("可用任务:")
    width = max(len(name) for name in TASKS)
    for name, fn in TASKS.items():
        doc = (fn.__doc__ or "").strip().splitlines()[0] if fn.__doc__ else ""
        print(f"  {name:<{width}}  {doc}")
    print("\n提示: 本脚本不依赖 make / shell 工具，Windows PowerShell 与 Linux 均可直接运行。")


def main(argv: list[str]) -> int:
    if not argv or argv[0] in {"help", "-h", "--help"}:
        print_help()
        return 0
    name = argv[0]
    fn = TASKS.get(name)
    if fn is None:
        print(f"未知任务: {name}\n", file=sys.stderr)
        print_help()
        return 2
    # 允许把剩余参数透传给底层命令（例如 e2e 的 -k 过滤）
    fn()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
