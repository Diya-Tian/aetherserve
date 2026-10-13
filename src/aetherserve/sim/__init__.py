"""性能仿真包：在不依赖 GPU 的前提下量化连续批处理的收益。"""

from .backend import (
    SerialReferenceBackend,
    SimulatedBackend,
    SimulatedHardware,
    reference_hardware,
)
from .runner import LatencyStats, LoadTestResult, make_prompt, run_load_test

__all__ = [
    "LatencyStats",
    "LoadTestResult",
    "SerialReferenceBackend",
    "SimulatedBackend",
    "SimulatedHardware",
    "make_prompt",
    "reference_hardware",
    "run_load_test",
]
