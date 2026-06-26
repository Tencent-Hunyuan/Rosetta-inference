"""ARC-Challenge metric.

Reuses the same logit-based 4-choice MC evaluation as MMLU.
"""
from evaluation.metrics.mmlu.metric import MMLUMetric


class ARCCMetric(MMLUMetric):
    """ARC-Challenge accuracy metric (logit-based, 4 choices A-D)."""

    def __init__(self, dataset_name="arc_challenge", **kwargs):
        super().__init__(dataset_name=dataset_name, **kwargs)

    def compute_metrics(self, results):
        out = super().compute_metrics(results)
        # Rename key for clarity in the JSON output
        return {"arc_challenge_" + k: v for k, v in out.items()}
