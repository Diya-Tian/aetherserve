"""pytest 共享夹具。"""

from __future__ import annotations

import asyncio
import os
import sys
import types
from typing import Any

import pytest

from aetherserve.backends.echo_backend import EchoBackend
from aetherserve.config import Settings
from aetherserve.engine import BatchPlan, Scheduler, build_engine
from aetherserve.engine.queue import AdmissionQueue


@pytest.fixture(autouse=True)
def _isolate_from_host_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """隔离宿主机环境：测试绝不发起真实网络请求。

    为什么必须做这件事？
    ----------------
    单元测试必须在任何机器上都得到相同结果，且不能依赖/污染外部环境。
    这里踩过两个坑：

    1. 开发机设置了 ``AETHERSERVE_PROMETHEUS_URL`` 后，
       ``_fetch_canary_metrics`` 会真的去请求该地址，导致测试既慢又随环境失败；
    2. 未打桩的 Kubernetes 客户端会尝试读 ``~/.kube/config`` 并联集群。

    实现注意：本夹具必须**在测试自己的 monkeypatch 之前**清掉变量，
    因此它显式接收 ``monkeypatch`` 并先删除，再由测试设置自己需要的值。
    （若测试发现变量被清掉，说明夹具顺序有问题 —— 本夹具的删除是"基线清理"，
     测试后续 setenv 仍然生效，因为同一个 monkeypatch 实例的 set 会覆盖 del。）
    """
    for var in (
        "AETHERSERVE_PROMETHEUS_URL",
        "AETHERSERVE_VLLM_ENDPOINT",
        "AETHERSERVE_API_KEYS",
        "KUBECONFIG",
    ):
        monkeypatch.delenv(var, raising=False)
    # 清掉所有 AETHERSERVE_* 变量，避免开发机上的配置影响测试
    for key in [k for k in os.environ if k.startswith("AETHERSERVE_")]:
        monkeypatch.delenv(key, raising=False)


@pytest.fixture(autouse=True)
def _block_real_kube_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """禁止测试真的去加载 kube 配置（未打桩时直接报错，暴露测试缺陷）。"""

    def _fail(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("测试中不应加载真实 kube 配置：请用 fake_k8s 夹具打桩 kubernetes 客户端")

    try:
        from kubernetes import config as kube_config
    except ImportError:  # pragma: no cover - 未安装 operator 依赖时跳过
        return
    monkeypatch.setattr(kube_config, "load_kube_config", _fail, raising=False)
    monkeypatch.setattr(kube_config, "load_incluster_config", _fail, raising=False)


@pytest.fixture
def fast_settings() -> Settings:
    """把批处理参数调小，让测试在毫秒级完成，同时更快触发边界条件。"""
    s = Settings()
    s.batching.max_batch_size = 4
    s.batching.max_batch_tokens = 512
    s.batching.max_wait_ms = 5.0
    s.batching.max_queue_size = 16
    s.batching.max_queue_tokens = 2048
    s.batching.worker_concurrency = 2
    s.backend.model = "echo-1"
    s.backend.backend = "echo"
    s.backend.default_max_tokens = 16
    s.environment = "test"
    return s


@pytest.fixture
def echo_backend() -> EchoBackend:
    # 去掉人为延迟，测试只关心逻辑正确性
    return EchoBackend(token_delay_ms=0.0, max_tokens_hint=8)


@pytest.fixture
def plan() -> BatchPlan:
    return BatchPlan(max_batch_size=4, max_batch_tokens=512, max_wait_ms=5.0, bucket_step=64)


@pytest.fixture
def queue() -> AdmissionQueue:
    return AdmissionQueue(max_size=8, max_tokens=1024, aging_ms=100.0)


@pytest.fixture
async def engine(fast_settings: Settings, echo_backend: EchoBackend):
    """已启动的引擎，测试结束自动优雅停机。"""
    eng = build_engine(fast_settings, echo_backend)
    await eng.start()
    try:
        yield eng
    finally:
        await eng.stop(drain=False)


@pytest.fixture
def scheduler(plan: BatchPlan) -> Scheduler:
    return Scheduler(plan, concurrency=2, kv_budget_tokens=1024)


# ===================================================================== 假 K8s 客户端
# 为什么放在 conftest？
# ------------------
# ``test_operator_handlers.py`` 与 ``test_operator_kopf.py`` 都需要它，
# 而 pytest 的夹具不跨测试文件共享（相对导入在 conftest 里也不可用）。
# 放在 conftest 后两个文件共用同一份实现，避免两份假客户端逐渐不一致。
#
# 为什么必须替换 sys.modules？
# --------------------------
# ``handlers`` 内部是函数级延迟导入（``from kubernetes import client``），
# 直接替换模块属性不会生效，必须替换 ``sys.modules`` 中的模块对象。


class FakeApiException(Exception):  # noqa: N818 - 刻意对齐真实客户端异常名，便于对照阅读
    """模仿 kubernetes.client.exceptions.ApiException。"""

    def __init__(self, status: int, reason: str = "") -> None:
        super().__init__(f"{status} {reason}")
        self.status = status
        self.reason = reason


class Recorded:
    """记录假客户端收到的每次调用。"""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def add(self, method: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        self.calls.append((method, args, kwargs))

    def methods(self) -> list[str]:
        return [c[0] for c in self.calls]

    def find(self, method: str) -> tuple[Any, ...] | None:
        for m, args, _ in self.calls:
            if m == method:
                return args
        return None


class _FakeResource:
    def __init__(self, recorder: Recorded, behavior: str) -> None:
        self.behavior = behavior

    def finish(self, method: str) -> Any:
        if self.behavior == "conflict" and method.startswith("create"):
            raise FakeApiException(409, "AlreadyExists")
        if self.behavior == "notfound" and method.startswith("delete"):
            raise FakeApiException(404, "NotFound")
        if self.behavior == "error" and method.startswith("create"):
            raise FakeApiException(500, "InternalError")
        return types.SimpleNamespace(metadata=types.SimpleNamespace(name="x"))


def _apps_api(recorder: Recorded, behavior: str) -> type:
    r = _FakeResource(recorder, behavior)

    class AppsV1Api:
        def create_namespaced_deployment(self, ns: str, body: Any) -> Any:
            recorder.add("create_namespaced_deployment", (ns, body), {})
            return r.finish("create")

        def patch_namespaced_deployment(self, name: str, ns: str, body: Any) -> Any:
            recorder.add("patch_namespaced_deployment", (name, ns, body), {})
            return r.finish("patch")

        def delete_namespaced_deployment(self, name: str, ns: str) -> Any:
            recorder.add("delete_namespaced_deployment", (name, ns), {})
            return r.finish("delete")

        def list_namespaced_deployment(self, ns: str, **kw: Any) -> Any:
            recorder.add("list_namespaced_deployment", (ns,), kw)
            return types.SimpleNamespace(items=[])

    return AppsV1Api


def _core_api(recorder: Recorded, behavior: str) -> type:
    r = _FakeResource(recorder, behavior)

    class CoreV1Api:
        def create_namespaced_service(self, ns: str, body: Any) -> Any:
            recorder.add("create_namespaced_service", (ns, body), {})
            return r.finish("create")

        def patch_namespaced_service(self, name: str, ns: str, body: Any) -> Any:
            recorder.add("patch_namespaced_service", (name, ns, body), {})
            return r.finish("patch")

        def delete_namespaced_service(self, name: str, ns: str) -> Any:
            recorder.add("delete_namespaced_service", (name, ns), {})
            return r.finish("delete")

        def list_namespaced_service(self, ns: str, **kw: Any) -> Any:
            recorder.add("list_namespaced_service", (ns,), kw)
            return types.SimpleNamespace(items=[])

    return CoreV1Api


def _autoscaling_api(recorder: Recorded, behavior: str) -> type:
    r = _FakeResource(recorder, behavior)

    class AutoscalingV2Api:
        def create_namespaced_horizontal_pod_autoscaler(self, ns: str, body: Any) -> Any:
            recorder.add("create_namespaced_horizontal_pod_autoscaler", (ns, body), {})
            return r.finish("create")

        def patch_namespaced_horizontal_pod_autoscaler(self, name: str, ns: str, body: Any) -> Any:
            recorder.add("patch_namespaced_horizontal_pod_autoscaler", (name, ns, body), {})
            return r.finish("patch")

        def delete_namespaced_horizontal_pod_autoscaler(self, name: str, ns: str) -> Any:
            recorder.add("delete_namespaced_horizontal_pod_autoscaler", (name, ns), {})
            return r.finish("delete")

        def list_namespaced_horizontal_pod_autoscaler(self, ns: str, **kw: Any) -> Any:
            recorder.add("list_namespaced_horizontal_pod_autoscaler", (ns,), kw)
            return types.SimpleNamespace(items=[])

    return AutoscalingV2Api


def _policy_api(recorder: Recorded, behavior: str) -> type:
    r = _FakeResource(recorder, behavior)

    class PolicyV1Api:
        def create_namespaced_pod_disruption_budget(self, ns: str, body: Any) -> Any:
            recorder.add("create_namespaced_pod_disruption_budget", (ns, body), {})
            return r.finish("create")

        def patch_namespaced_pod_disruption_budget(self, name: str, ns: str, body: Any) -> Any:
            recorder.add("patch_namespaced_pod_disruption_budget", (name, ns, body), {})
            return r.finish("patch")

        def delete_namespaced_pod_disruption_budget(self, name: str, ns: str) -> Any:
            recorder.add("delete_namespaced_pod_disruption_budget", (name, ns), {})
            return r.finish("delete")

        def list_namespaced_pod_disruption_budget(self, ns: str, **kw: Any) -> Any:
            recorder.add("list_namespaced_pod_disruption_budget", (ns,), kw)
            return types.SimpleNamespace(items=[])

    return PolicyV1Api


class _FakeDynamicResource:
    def __init__(self, state: dict[str, Any]) -> None:
        self.state = state

    def create(self, body: Any, namespace: str) -> Any:
        self.state["recorder"].add("dynamic_create", (body, namespace), {})
        if self.state["behavior"] == "conflict":
            raise FakeApiException(409, "AlreadyExists")
        return body

    def patch(self, body: Any, name: str, namespace: str, content_type: str = "") -> Any:
        self.state["recorder"].add("dynamic_patch", (body, name, namespace), {"content_type": content_type})
        return body


class _FakeDynamicClient:
    def __init__(self, _api_client: Any) -> None:
        self._state = _ACTIVE_STATE["state"]

    @property
    def resources(self) -> Any:
        state = self._state

        class Registry:
            def get(self, api_version: str, kind: str) -> _FakeDynamicResource:
                state["recorder"].add("resources_get", (api_version, kind), {})
                return _FakeDynamicResource(state)

        return Registry()


class _FakeApiClient:
    """只实现被用到的两个能力：序列化与反序列化。

    ``sanitize_for_serialization`` 刻意做成**递归转换**：
    手写少量字段的假实现会漏掉 labels 之类内容，
    让测试失败原因变成"假数据不全"而非真实代码缺陷（这是实际踩过的坑）。
    """

    def sanitize_for_serialization(self, obj: Any) -> Any:
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, dict):
            return {k: self.sanitize_for_serialization(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self.sanitize_for_serialization(v) for v in obj]
        attrs = getattr(obj, "__dict__", None)
        if attrs is not None:
            return {k: self.sanitize_for_serialization(v) for k, v in attrs.items() if not k.startswith("_")}
        return str(obj)

    def deserialize(self, manifest: dict[str, Any], model: str) -> Any:
        return types.SimpleNamespace(**manifest)

    def _ApiClient__deserialize(self, manifest: dict[str, Any], model: str) -> Any:  # noqa: N802 - 名称改写后的私有方法，必须保持一致
        return types.SimpleNamespace(**manifest)


# 当前激活的夹具状态（供假动态客户端读取）
_ACTIVE_STATE: dict[str, Any] = {"state": {"behavior": "created", "recorder": Recorded()}}


@pytest.fixture
def fake_k8s(monkeypatch: pytest.MonkeyPatch):
    """注入假的 ``kubernetes.client`` / ``kubernetes.dynamic`` 模块。

    Returns:
        ``(recorder, state)``；修改 ``state["behavior"]`` 可切换
        ``created`` / ``conflict`` / ``notfound`` / ``error`` 行为。
    """
    recorder = Recorded()
    state: dict[str, Any] = {"behavior": "created", "recorder": recorder}
    _ACTIVE_STATE["state"] = state

    fake_client = types.ModuleType("kubernetes.client")
    fake_client.ApiClient = _FakeApiClient
    fake_client.ApiException = FakeApiException

    def _ctor(factory):  # type: ignore[no-untyped-def]
        def build(*args: Any, **kwargs: Any) -> Any:
            return factory(recorder, state["behavior"])()

        return build

    fake_client.AppsV1Api = _ctor(_apps_api)
    fake_client.CoreV1Api = _ctor(_core_api)
    fake_client.AutoscalingV2Api = _ctor(_autoscaling_api)
    fake_client.PolicyV1Api = _ctor(_policy_api)

    exceptions_mod = types.ModuleType("kubernetes.client.exceptions")
    exceptions_mod.ApiException = FakeApiException
    fake_client.exceptions = exceptions_mod

    fake_dynamic = types.ModuleType("kubernetes.dynamic")
    fake_dynamic.DynamicClient = _FakeDynamicClient

    fake_kubernetes = types.ModuleType("kubernetes")
    fake_kubernetes.client = fake_client
    fake_kubernetes.dynamic = fake_dynamic

    monkeypatch.setitem(sys.modules, "kubernetes", fake_kubernetes)
    monkeypatch.setitem(sys.modules, "kubernetes.client", fake_client)
    monkeypatch.setitem(sys.modules, "kubernetes.client.exceptions", exceptions_mod)
    monkeypatch.setitem(sys.modules, "kubernetes.dynamic", fake_dynamic)

    yield recorder, state


def pytest_configure(config: pytest.Config) -> None:
    """确保事件循环策略在 Windows 上使用 Selector（兼容性更好）。"""
    if hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):  # pragma: no cover - 仅 Windows
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
