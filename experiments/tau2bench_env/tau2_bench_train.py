"""Training wrapper around primeintellect/tau2-bench.

The upstream `Tau2BenchEnv` only sets `eval_dataset` because it ships as a
benchmark / eval harness. Verifiers' `Environment.get_dataset()` reads
`self.dataset` and raises `ValueError("dataset is not set")` when used as
a training source — which is exactly what the prime-rl orchestrator does.

This wrapper subclasses `Tau2BenchEnv` and mirrors `eval_dataset_source`
into `dataset_source` so the same task pool is available for both training
and evaluation. Compaction / padding monkey-patches in
`src/kv_eviction/env.py` apply transparently.
"""

from typing import Any

import verifiers as vf
from tau2_bench import Tau2BenchEnv


class Tau2BenchTrainEnv(Tau2BenchEnv):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        if self.dataset_source is None and self.eval_dataset_source is not None:
            self.dataset_source = self.eval_dataset_source


def load_environment(**kwargs: Any) -> vf.Environment:
    """Entry point for `vf.load_environment("tau2-bench-train", ...)`.

    All kwargs are forwarded to `Tau2BenchEnv.__init__` (domain, user_model,
    user_args, user_base_url, user_api_key_var, max_steps, max_errors,
    max_workers, max_turns).
    """
    return Tau2BenchTrainEnv(**kwargs)
