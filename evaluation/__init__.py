"""
SWE-Bench Evaluation Module

Provides Docker-based evaluation for SWE-Bench instances,
matching the official SWE-Bench Pro evaluation setup.

Main components:
- docker_runner: Docker container management and execution
- evaluator: Evaluation orchestration
- metrics: Metrics computation and reporting
"""

__version__ = "0.1.0"

from .docker_runner import DockerRunner, EvaluationResult, SWEBenchInstance
from .evaluator import SWEBenchEvaluator, EvaluationConfig, evaluate, compare_models
from .metrics import compute_metrics, MetricsReport

__all__ = [
    "DockerRunner",
    "EvaluationResult",
    "SWEBenchInstance",
    "SWEBenchEvaluator",
    "EvaluationConfig",
    "evaluate",
    "compare_models",
    "compute_metrics",
    "MetricsReport",
]
