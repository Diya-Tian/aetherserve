"""多租户鉴权：API-Key → 租户身份。

为什么用 API-Key 而不是 JWT？
----------------------------
推理服务的调用方是**服务端程序**（不是浏览器），
鉴权目标是"识别租户 + 便于按租户配额与计费"，不需要无状态会话与短期过期。
API-Key 结构简单、可缓存、可在网关做常数时间比对，
且天然适配"按 Key 限流/计费"的网关模型。
若要与企业的统一身份体系对接，只需把 ``ApiKeyAuthenticator`` 换成一个
调用 OIDC/JWKS 的验证器 —— 因为网关只依赖 ``Principal`` 这个身份抽象。
"""

from __future__ import annotations

import hmac
from dataclasses import dataclass

from ..observability.logging import get_logger

log = get_logger(__name__)


@dataclass(slots=True, frozen=True)
class Principal:
    """通过鉴权后的调用者身份。"""

    api_key: str
    tenant: str
    rps: float
    burst: int

    @property
    def tenant_key(self) -> str:
        return self.tenant


class AuthError(Exception):
    """鉴权失败（401）。"""


class ApiKeyAuthenticator:
    """基于内存映射的 API-Key 校验器。

    安全说明：真实生产应存储 Key 的**哈希**（如 SHA-256 + 盐），
    并支持轮换与吊销。这里为便于演示保存明文，但比对使用
    ``hmac.compare_digest`` 做常数时间比较，避免时序侧信道。
    """

    def __init__(self, keys: dict[str, tuple[str, float, int]]) -> None:
        self._keys = dict(keys)

    def add(self, api_key: str, tenant: str, rps: float, burst: int) -> None:
        self._keys[api_key] = (tenant, rps, burst)

    @property
    def key_count(self) -> int:
        return len(self._keys)

    def authenticate(self, api_key: str | None) -> Principal:
        """校验 API-Key。

        Raises:
            AuthError: Key 缺失或不匹配。
        """
        if not api_key:
            raise AuthError("缺少 API-Key，请在 Authorization: Bearer <key> 中提供")

        for known, (tenant, rps, burst) in self._keys.items():
            if hmac.compare_digest(known, api_key):
                return Principal(api_key=api_key, tenant=tenant, rps=rps, burst=burst)

        log.warning("auth_failed", key_prefix=api_key[:8])
        raise AuthError("API-Key 无效")
