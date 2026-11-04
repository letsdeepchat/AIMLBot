"""Quest copilot ML core: profiling, leakage, shift, validation design and baselines.

Pure Python with no web dependencies, so it runs the same from the API worker, a notebook or tests.
"""
from .experiments import build_leaderboard, run_experiment
from .pipeline import run_analysis, run_baselines
from .spec import METRICS, ChallengeSpec

__all__ = ["ChallengeSpec", "METRICS", "build_leaderboard", "run_analysis", "run_baselines", "run_experiment"]
