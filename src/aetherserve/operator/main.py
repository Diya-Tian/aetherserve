"""Operator 进程入口。

用法::

    # 本地开发（读取 ~/.kube/config）
    python -m aetherserve.operator.main

    # 集群内（使用 ServiceAccount）
    python -m aetherserve.operator.main --verbose

kopf 会自动处理：watch 流、断线重连、事件去重、指数退避重试。
"""

from __future__ import annotations

import logging
import sys


def main(argv: list[str] | None = None) -> int:
    import kopf

    from .handlers import register_handlers

    logging.basicConfig(
        level=logging.DEBUG
        if (argv or sys.argv[1:] or []) and "--verbose" in (argv or sys.argv[1:])
        else logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    register_handlers()
    kopf.run(
        # 只处理带这张标签的 CR，避免与其他 Operator 抢同一类资源
        clusterwide=False,
        liveness_endpoint="http://0.0.0.0:8080/healthz",
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
