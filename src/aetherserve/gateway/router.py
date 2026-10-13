"""模型路由与负载均衡。

当前实现：单后端 + 熔断保护（本地开发与压测足够）。
设计已为多实例预留扩展点：``ModelRoute.targets`` 是一个后端列表，
``pick_target`` 用**一致性哈希**在请求维度做粘性路由 —— 选它而非轮询的理由是
**KV Cache 亲和性**：同一会话（相同 messages 前缀）落到同一实例，
可复用已计算的 prefix cache（vLLM 的 Automatic Prefix Caching），
从而把长 system prompt 场景的首 token 延迟降低数倍。
若实例集合变化导致哈希环变化，普通请求只是失去缓存收益，不会失败；
只有正在处理中的流式请求会受影响，因此实践上配合"优雅摘除"（先停止新流量再下线实例）。
"""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass, field
from typing import Any

from ..backends.base import InferenceBackend
from ..observability.logging import get_logger
from .circuit import CircuitBreakerRegistry, CircuitOpenError

log = get_logger(__name__)


@dataclass
class ModelRoute:
    """一个逻辑模型到物理后端的映射。"""

    model: str
    targets: list[InferenceBackend] = field(default_factory=list)
    weights: list[int] = field(default_factory=list)
    # 会话粘性开关：关闭后退化为轮询
    sticky: bool = True


class BackendRouter:
    """按模型名路由请求，并提供熔断保护。"""

    def __init__(self, breakers: CircuitBreakerRegistry | None = None) -> None:
        self._routes: dict[str, ModelRoute] = {}
        self._default_model: str | None = None
        self.breakers = breakers or CircuitBreakerRegistry()
        self._rr_counter = 0
        self._lock = asyncio.Lock()
        self._hash_ring: dict[str, list[tuple[int, InferenceBackend]]] = {}

    # ------------------------------------------------------------ 注册
    def register(self, model: str, backend: InferenceBackend, weights: int = 1, sticky: bool = True) -> None:
        """注册（或追加）一个模型的后端实例。"""
        route = self._routes.get(model)
        if route is None:
            route = ModelRoute(model=model, sticky=sticky)
            self._routes[model] = route
            if self._default_model is None:
                self._default_model = model
        route.targets.append(backend)
        route.weights.append(max(weights, 1))
        self._build_ring(model)

    def set_default(self, model: str) -> None:
        if model not in self._routes:
            raise KeyError(f"模型 {model} 未注册")
        self._default_model = model

    def _build_ring(self, model: str) -> None:
        """构造一致性哈希环（虚拟节点数 = 权重 × 64）。"""
        route = self._routes[model]
        ring: list[tuple[int, InferenceBackend]] = []
        for idx, backend in enumerate(route.targets):
            vnodes = route.weights[idx] * 64
            for v in range(vnodes):
                key = f"{model}#{backend.name}#{idx}#{v}".encode()
                ring.append((int.from_bytes(hashlib.md5(key).digest()[:4], "big"), backend))
        ring.sort(key=lambda x: x[0])
        self._hash_ring[model] = ring

    # ------------------------------------------------------------ 查询
    @property
    def models(self) -> list[str]:
        return sorted(self._routes)

    @property
    def default_model(self) -> str | None:
        return self._default_model

    def get_route(self, model: str | None) -> ModelRoute:
        """解析模型名；未指定时回退默认模型。

        Raises:
            KeyError: 模型未注册（网关应返回 404 model_not_found）。
        """
        name = model or self._default_model
        if name is None:
            raise KeyError("网关未注册任何模型")
        route = self._routes.get(name)
        if route is None:
            raise KeyError(f"模型 {name!r} 不存在，可用模型：{', '.join(self.models)}")
        return route

    def pick_target(self, model: str, session_key: str | None = None) -> InferenceBackend:
        """选择一个后端实例。

        Args:
            model: 逻辑模型名。
            session_key: 会话标识（通常取 messages 的哈希）；为 None 时轮询。
        """
        route = self.get_route(model)
        if not route.targets:
            raise KeyError(f"模型 {model} 没有可用后端")

        if not route.sticky or session_key is None or len(route.targets) == 1:
            self._rr_counter += 1
            return route.targets[self._rr_counter % len(route.targets)]

        ring = self._hash_ring.get(model) or []
        if not ring:
            return route.targets[0]
        h = int.from_bytes(hashlib.md5(session_key.encode()).digest()[:4], "big")
        # 二分查找环上第一个 >= h 的虚拟节点
        lo, hi = 0, len(ring)
        while lo < hi:
            mid = (lo + hi) // 2
            if ring[mid][0] < h:
                lo = mid + 1
            else:
                hi = mid
        return ring[lo % len(ring)][1]

    def breaker_for(self, backend_name: str) -> Any:
        return self.breakers.get(backend_name)

    async def health(self) -> dict[str, Any]:
        """聚合所有后端健康状态（/healthz 与 Operator 探针使用）。"""
        out: dict[str, Any] = {}
        for model, route in self._routes.items():
            states = []
            for backend in route.targets:
                try:
                    states.append(await asyncio.wait_for(backend.health(), timeout=3.0))
                except Exception as exc:
                    states.append({"backend": backend.name, "ready": False, "error": str(exc)})
            out[model] = {"targets": states, "ready": all(s.get("ready") for s in states)}
        return out


__all__ = ["BackendRouter", "CircuitOpenError", "ModelRoute"]
