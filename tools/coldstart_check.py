"""冷启动端到端验证：模拟评审者 clone 仓库后按 README 操作。

为什么必须做这一步？
-------------------
项目文档写了"克隆下来就能跑"，但**作者本机的环境可能掩盖依赖缺失**。
最可靠的验证是：在一个**全新的虚拟环境**里，只按 README 的步骤操作，
看能否跑通"安装 → 起服务 → 调用流式接口 → 抓指标"。

本脚本做的事：
1. 在临时目录建全新 venv（隔离本机已装的任何东西）
2. 用仓库里的 ``pyproject.toml`` 安装（等价于 ``pip install -e ".[dev]"``）
3. 启动网关（echo 后端，无需 GPU）
4. 依次验证 /healthz、/v1/models、流式 /v1/chat/completions、/metrics
5. 校验 SSE 帧格式与关键指标是否存在

⚠️ 注意：这一步会真的创建 venv 并安装依赖，耗时约 1-3 分钟。
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PORT = 18400
BASE = f"http://127.0.0.1:{PORT}"


def run(cmd: list[str], cwd: Path, timeout: int = 600) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )


def http_get(path: str, timeout: float = 10.0) -> tuple[int, str]:
    req = urllib.request.Request(f"{BASE}{path}", headers={"Authorization": "Bearer sk-demo-001"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")
    except Exception as e:
        return 0, str(e)


def http_post(path: str, payload: dict, timeout: float = 30.0) -> tuple[int, str]:
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=json.dumps(payload).encode(),
        headers={"Authorization": "Bearer sk-demo-001", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")
    except Exception as e:
        return 0, str(e)


def main() -> int:
    checks: list[tuple[str, bool, str]] = []

    def record(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, ok, detail))
        print(f"  [{'OK  ' if ok else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")

    workdir = Path(tempfile.mkdtemp(prefix="aetherserve-coldstart-"))
    venv_dir = workdir / "venv"
    print("=" * 78)
    print("冷启动端到端验证（全新 venv + 按 README 操作）")
    print("=" * 78)
    print(f"临时目录: {workdir}\n")

    proc: subprocess.Popen[str] | None = None
    try:
        # ---- 1) 建全新 venv
        print("[1] 创建全新虚拟环境")
        r = run([sys.executable, "-m", "venv", str(venv_dir)], cwd=workdir, timeout=180)
        record("venv 创建", r.returncode == 0, r.stderr.strip()[:120] if r.returncode else "")
        if r.returncode != 0:
            return 1

        py = venv_dir / ("Scripts" if sys.platform == "win32" else "bin") / "python"
        py = py.with_suffix(".exe") if sys.platform == "win32" else py

        # ---- 2) 按 README 安装
        print("\n[2] 安装依赖（pip install -e '.[dev]'）")
        r = run([str(py), "-m", "pip", "install", "-q", "-e", ".[dev]"], cwd=REPO_ROOT, timeout=900)
        record("依赖安装", r.returncode == 0, (r.stderr or "").strip()[-200:] if r.returncode else "")

        # ---- 3) 验证 console script 可用（pyproject 声明的入口）
        print("\n[3] 校验包入口与导入链")
        r = run(
            [str(py), "-c", "import aetherserve; print(aetherserve.__version__)"], cwd=workdir, timeout=120
        )
        record("包可导入", r.returncode == 0, r.stdout.strip() or r.stderr.strip()[:120])

        r = run(
            [str(py), "-c", "from aetherserve.gateway.app import create_app; print('ok')"],
            cwd=workdir,
            timeout=120,
        )
        record("应用可构造", r.returncode == 0, r.stdout.strip() or r.stderr.strip()[:160])

        # ---- 4) 启动网关（echo 后端）
        print("\n[4] 启动网关（echo 后端，无需 GPU）")
        proc = subprocess.Popen(
            [
                str(py),
                "-m",
                "uvicorn",
                "aetherserve.gateway.app:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(PORT),
            ],
            cwd=workdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        ready = False
        for _ in range(40):
            time.sleep(1)
            code, _body = http_get("/healthz", timeout=3)
            if code == 200:
                ready = True
                break
            if proc.poll() is not None:
                break
        record("服务就绪（/healthz 200）", ready)
        if not ready:
            if proc.poll() is not None and proc.stdout is not None:
                print("    启动输出:", proc.stdout.read()[-400:])
            return 1

        # ---- 5) 健康与模型列表
        print("\n[5] 健康检查与模型列表")
        code, body = http_get("/healthz")
        record("/healthz 返回 200", code == 200, body[:80])
        record("engine_running 为真", '"engine_running":true' in body.replace(" ", ""))

        code, body = http_get("/v1/models")
        record("/v1/models 返回 200", code == 200, body[:80])
        record("列出 echo-1 模型", "echo-1" in body)

        # ---- 6) 鉴权必须生效
        print("\n[6] 鉴权校验（无 key 应 401）")
        req = urllib.request.Request(f"{BASE}/v1/models")
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                record("无 key 被拒绝", resp.status == 401, f"实际 {resp.status}")
        except urllib.error.HTTPError as e:
            record("无 key 被拒绝（401）", e.code == 401, f"实际 {e.code}")

        # ---- 7) 非流式
        print("\n[7] 非流式对话补全")
        code, body = http_post(
            "/v1/chat/completions",
            {"model": "echo-1", "messages": [{"role": "user", "content": "你好"}], "max_tokens": 6},
        )
        ok = code == 200
        record("非流式返回 200", ok, body[:80] if not ok else "")
        if ok:
            payload = json.loads(body)
            record("响应结构符合 OpenAI 协议", payload.get("object") == "chat.completion")
            record("含 choices 与 usage", bool(payload.get("choices")) and "usage" in payload)

        # ---- 8) 流式（SSE）
        print("\n[8] 流式对话补全（SSE）")
        code, body = http_post(
            "/v1/chat/completions",
            {
                "model": "echo-1",
                "messages": [{"role": "user", "content": "你好"}],
                "max_tokens": 5,
                "stream": True,
            },
        )
        record("流式返回 200", code == 200, body[:120] if code != 200 else "")
        record("含 data: 帧", "data: " in body)
        record("以 [DONE] 结束", body.rstrip().endswith("[DONE]"))
        frames = [ln for ln in body.splitlines() if ln.startswith("data: ") and "[DONE]" not in ln]
        record("收到多个增量帧", len(frames) >= 2, f"{len(frames)} 帧")
        record("帧均为单行 JSON", all(_is_single_line_json(f) for f in frames))

        # ---- 9) 指标
        print("\n[9] Prometheus 指标")
        req = urllib.request.Request(f"{BASE}/metrics")
        with urllib.request.urlopen(req, timeout=10) as resp:
            metrics = resp.read().decode()
        for name in (
            "aetherserve_requests_total",
            "aetherserve_ttft_seconds",
            "aetherserve_batch_size",
            "aetherserve_queue_depth",
            "aetherserve_tokens_total",
        ):
            record(f"指标 {name} 已暴露", name in metrics)

        # ---- 10) 批量请求（验证批处理链路不崩）
        print("\n[10] 并发请求（验证批处理链路）")
        import concurrent.futures

        def call(i: int) -> int:
            c, _ = http_post(
                "/v1/chat/completions",
                {"model": "echo-1", "messages": [{"role": "user", "content": f"q{i}"}], "max_tokens": 4},
            )
            return c

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            codes = list(pool.map(call, range(12)))
        record("12 个并发请求全部成功", all(c == 200 for c in codes), f"状态码 {sorted(set(codes))}")

    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
        shutil.rmtree(workdir, ignore_errors=True)

    # ---- 汇总
    print()
    print("=" * 78)
    passed = sum(1 for _n, ok, _d in checks if ok)
    failed = [n for n, ok, _d in checks if not ok]
    print(f"结果: {passed}/{len(checks)} 项通过")
    if failed:
        print("未通过项:")
        for name in failed:
            print(f"  - {name}")
    print("=" * 78)
    return 1 if failed else 0


def _is_single_line_json(frame: str) -> bool:
    payload = frame[len("data: ") :].strip()
    try:
        json.loads(payload)
    except json.JSONDecodeError:
        return False
    return "\n" not in payload


if __name__ == "__main__":
    raise SystemExit(main())
