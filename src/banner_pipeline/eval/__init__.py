"""Multi-region evaluation framework for virtual banner placement.

See docs/EVALUATION.md for the contract.

Public API: run_eval(experiment_dir, reference_dir=None, ...) -> dict
"""

from banner_pipeline.eval.runner import run_eval

__all__ = ["run_eval"]
