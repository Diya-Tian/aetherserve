"""端到端测试（需要真实 kind 集群）。

运行方式::

    make cluster          # 创建 kind 集群
    make crd              # 安装 CRD
    make e2e              # 运行本目录下的测试

无集群时会发生什么
----------------
本目录所有测试都会 **skip**（而不是报错）。判断依据：
``kubectl version`` 能否在 5 秒内连上 API Server。
这样开发机（无集群）跑 ``make test`` 不会失败，
而有集群时又能得到真实的端到端验证。

为什么要单独放 e2e 而不是塞进单测
-------------------------------
单测用假客户端验证"代码逻辑是否正确"（快、可靠、可离线）；
e2e 验证"**真实 API Server 是否接受我们生成的清单**"——
这是假客户端永远无法覆盖的一类风险（字段拼错、类型不对、
不可变字段冲突、CRD schema 校验失败）。两者互补，缺一不可。
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
import uuid
from collections.abc import Iterator
from typing import Any

import pytest

pytestmark = pytest.mark.e2e

NAMESPACE = "aetherserve-e2e"
CRD_NAME = "inferenceservices.aetherserve.io"
KUBECTL_TIMEOUT = 30


def kubectl_available() -> bool:
    """kubectl 是否存在。"""
    return shutil.which("kubectl") is not None


def cluster_reachable() -> bool:
    """集群是否可达（5 秒内能拿到版本）。"""
    if not kubectl_available():
        return False
    try:
        r = subprocess.run(
            ["kubectl", "version", "-o", "json", "--request-timeout=5s"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    return r.returncode == 0


def kube(*args: str, timeout: int = KUBECTL_TIMEOUT, check: bool = False) -> subprocess.CompletedProcess[str]:
    """执行 kubectl 并返回结果（不抛异常，由调用方判断 returncode）。"""
    return subprocess.run(
        ["kubectl", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=check,
    )


def kube_json(*args: str) -> Any:
    r = kube(*args)
    if r.returncode != 0:
        raise AssertionError(f"kubectl {' '.join(args)} 失败: {r.stderr.strip()}")
    return json.loads(r.stdout)


@pytest.fixture(scope="session", autouse=True)
def require_cluster() -> None:
    """无集群时跳过整个 e2e 目录。"""
    if not kubectl_available():
        pytest.skip("kubectl 未安装 —— 跳过端到端测试（安装后可用 make cluster 创建集群）")
    if not cluster_reachable():
        pytest.skip("Kubernetes 集群不可达 —— 跳过端到端测试（见 docs/部署与本地验证.md）")


@pytest.fixture(scope="session")
def crd_installed() -> None:
    """确认 InferenceService CRD 已安装；未安装则跳过（而不是失败）。"""
    r = kube("get", "crd", CRD_NAME)
    if r.returncode != 0:
        pytest.skip(f"CRD {CRD_NAME} 未安装 —— 先执行 make crd（见 docs/部署与本地验证.md）")


@pytest.fixture(scope="session")
def namespace(crd_installed: None) -> Iterator[str]:
    """创建专用命名空间，测试结束删除。"""
    kube("create", "namespace", NAMESPACE)
    try:
        yield NAMESPACE
    finally:
        kube("delete", "namespace", NAMESPACE, "--wait=false", timeout=60)


def make_isvc(name: str, **spec: Any) -> dict[str, Any]:
    """构造一个用于测试的 InferenceService 对象。"""
    body: dict[str, Any] = {
        "apiVersion": "aetherserve.io/v1alpha1",
        "kind": "InferenceService",
        "metadata": {"name": name, "namespace": NAMESPACE},
        "spec": {"model": "e2e-test-model", "backend": "echo", "replicas": 1, **spec},
    }
    return body


def apply(body: dict[str, Any], timeout: int = KUBECTL_TIMEOUT) -> subprocess.CompletedProcess[str]:
    """通过 stdin 应用一个对象（避免临时文件；且能精确拿到 stderr）。"""
    return subprocess.run(
        ["kubectl", "apply", "-f", "-"],
        input=json.dumps(body),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )


def delete_isvc(name: str, wait: bool = False) -> None:
    args = ["delete", "isvc", name, "-n", NAMESPACE, "--ignore-not-found"]
    if wait:
        args.append("--wait=true")
    kube(*args, timeout=60)


def wait_for(desc: str, condition: Any, timeout_s: float = 90.0, interval_s: float = 3.0) -> None:
    """轮询等待条件成立，超时给出可读的失败信息。"""
    deadline = time.monotonic() + timeout_s
    last = ""
    while time.monotonic() < deadline:
        ok, detail = condition()
        last = detail
        if ok:
            return
        time.sleep(interval_s)
    pytest.fail(f"等待「{desc}」超时（{timeout_s}s）。最后观测：{last}")


class TestClusterAndCrd:
    """集群与 CRD 层面的验证。"""

    def test_crd_passes_server_side_validation(self, crd_installed: None) -> None:
        """CRD 在 API Server 里是 Established 状态。

        这一步能抓到"清单结构合法但 schema 不被接受"的问题 ——
        假客户端（单测）无法覆盖这类风险。
        """
        crd = kube_json("get", "crd", CRD_NAME, "-o", "json")
        conditions = {c["type"]: c["status"] for c in crd["status"]["conditions"]}
        assert conditions.get("Established") == "True", f"CRD 未 Established: {conditions}"

    def test_crd_declares_printer_columns(self, crd_installed: None) -> None:
        """自定义列已生效 —— 决定 kubectl get isvc 的可读性。"""
        out = kube(
            "get", "crd", CRD_NAME, "-o", "jsonpath={.spec.versions[0].additionalPrinterColumns[*].name}"
        )
        names = set(out.stdout.split())
        assert {"Model", "Backend", "Replicas", "Ready", "Canary"} <= names, f"缺少自定义列: {names}"

    def test_crd_schema_rejects_invalid_backend(self, crd_installed: None) -> None:
        """CRD schema 必须拒绝非法的 backend 取值（enum 校验生效）。

        这是"代码与 CRD 契约一致"的反向验证：如果 schema 没写 enum，
        用户就能创建出 Operator 无法处理的对象。
        """
        bad = make_isvc("e2e-invalid")
        bad["spec"]["backend"] = "not-a-real-backend"
        r = apply(bad)
        assert r.returncode != 0, "API Server 竟然接受了非法 backend —— CRD 的 enum 校验未生效"
        delete_isvc("e2e-invalid")

    def test_crd_defaults_and_validation_of_model(self, crd_installed: None) -> None:
        """spec.model 是必填项，缺失时必须被拒绝。"""
        bad = {
            "apiVersion": "aetherserve.io/v1alpha1",
            "kind": "InferenceService",
            "metadata": {"name": "e2e-no-model", "namespace": NAMESPACE},
            "spec": {"backend": "echo"},
        }
        r = apply(bad)
        assert r.returncode != 0, "API Server 接受了缺少 model 的对象 —— required 校验未生效"
        delete_isvc("e2e-no-model")


class TestInferenceServiceLifecycle:
    """若 Operator 在运行，验证它是否真的把资源调和出来。"""

    def _operator_running(self) -> bool:
        r = kube(
            "get",
            "pods",
            "-A",
            "-l",
            "app.kubernetes.io/name=aetherserve,app.kubernetes.io/component=operator",
            "-o",
            "jsonpath={.items[*].status.phase}",
        )
        return "Running" in (r.stdout or "")

    def test_isvc_accepted_by_api_server(self, namespace: str) -> None:
        """一个合法的 InferenceService 必须能创建成功。"""
        name = f"e2e-ok-{uuid.uuid4().hex[:6]}"
        r = apply(make_isvc(name))
        assert r.returncode == 0, f"创建 InferenceService 失败: {r.stderr.strip()}"
        try:
            obj = kube_json("get", "isvc", name, "-n", NAMESPACE, "-o", "json")
            assert obj["spec"]["model"] == "e2e-test-model"
        finally:
            delete_isvc(name)

    def test_operator_reconciles_resources(self, namespace: str) -> None:
        """若 Operator 在运行，应生成 Deployment 与 Service。

        未运行 Operator 时跳过（而不是失败）—— 因为"只装 CRD 不装 Operator"
        也是一种合法的实验形态，见 docs/部署与本地验证.md。
        """
        if not self._operator_running():
            pytest.skip("Operator 未运行 —— 先 make operator 或 make deploy 后再验证调和结果")

        name = f"e2e-recon-{uuid.uuid4().hex[:6]}"
        r = apply(make_isvc(name, replicas=1, serviceMonitor=False))
        assert r.returncode == 0, r.stderr
        try:
            wait_for(
                "Deployment 被创建",
                lambda: (
                    kube("get", "deployment", f"{name}-gateway", "-n", NAMESPACE).returncode == 0,
                    kube("get", "deployment", "-n", NAMESPACE).stdout.strip(),
                ),
                timeout_s=90,
            )
            wait_for(
                "Service 被创建",
                lambda: (
                    kube("get", "service", f"{name}-stable", "-n", NAMESPACE).returncode == 0,
                    kube("get", "service", "-n", NAMESPACE).stdout.strip(),
                ),
                timeout_s=60,
            )

            # 关键不变量：Service 必须真的能选中 Pod（selector 与 pod 标签一致）
            # 这是单测里用假客户端验证过的同一条不变量，这里在真实 API Server 上再验一次。
            dep = kube_json("get", "deployment", f"{name}-gateway", "-n", NAMESPACE, "-o", "json")
            pod_labels = dep["spec"]["template"]["metadata"]["labels"]
            svc = kube_json("get", "service", f"{name}-stable", "-n", NAMESPACE, "-o", "json")
            for k, v in svc["spec"]["selector"].items():
                assert pod_labels.get(k) == v, (
                    f"Service selector 与 Pod 标签不一致（{k}: {v} vs {pod_labels.get(k)}）—— 服务将选不到 Pod"
                )
        finally:
            delete_isvc(name)

    def test_deletion_cascades_to_children(self, namespace: str) -> None:
        """删除 CR 后，子资源必须被级联回收（验证 ownerReference 生效）。"""
        if not self._operator_running():
            pytest.skip("Operator 未运行 —— 无法验证级联回收")

        name = f"e2e-cascade-{uuid.uuid4().hex[:6]}"
        r = apply(make_isvc(name, serviceMonitor=False))
        assert r.returncode == 0, r.stderr
        wait_for(
            "Deployment 被创建",
            lambda: (
                kube("get", "deployment", f"{name}-gateway", "-n", NAMESPACE).returncode == 0,
                "等待中",
            ),
            timeout_s=90,
        )

        delete_isvc(name, wait=True)
        wait_for(
            "Deployment 被级联回收",
            lambda: (
                kube("get", "deployment", f"{name}-gateway", "-n", NAMESPACE).returncode != 0,
                "Deployment 仍然存在 —— ownerReference 可能未正确设置",
            ),
            timeout_s=90,
        )
