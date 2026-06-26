"""Big Bench Hard (BBH) metric.

Exact-match after light normalization: strip whitespace, lowercase,
remove surrounding parentheses (e.g. '(A)' → 'a').
Reports overall accuracy and per-task breakdown.
"""
import re
import pandas as pd

from evaluation.metrics.base_metric import BaseMetric


def _normalize(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip().lower()
    # Remove surrounding parentheses: (a) → a
    text = re.sub(r"^\((.+)\)$", r"\1", text)
    return text.strip()


def _extract_prediction(response: str) -> str:
    """Take the first non-empty line of the response as the prediction."""
    if not isinstance(response, str):
        return ""
    for line in response.splitlines():
        line = line.strip()
        if line:
            return line
    return response.strip()


class BBHMetric(BaseMetric):
    """BBH exact-match accuracy with per-task breakdown."""

    def __init__(self, dataset_name="bbh", report_by_task=True, **kwargs):
        super().__init__()
        self.dataset_name  = dataset_name
        self.report_by_task = report_by_task
        self.results = []

    def load_model(self, logger=None):
        pass

    def process(self, answers, labels, subjects=None, **kwargs):
        if answers is None:
            return self.results
        subjects = subjects or ["bbh"] * len(answers)
        for pred_text, label, task in zip(answers, labels, subjects):
            pred  = _normalize(_extract_prediction(pred_text))
            label = _normalize(str(label))
            self.results.append({
                "task":     task,
                "label":    label,
                "pred":     pred,
                "response": pred_text,
                "correct":  pred == label,
            })
        return self.results

    def compute_metrics(self, results):
        df = pd.DataFrame(results)
        out = {}

        if self.report_by_task:
            for task, group in sorted(df.groupby("task")):
                acc = group["correct"].mean()
                out[task] = (round(float(acc), 6), len(group))

        count    = len(df)
        avg_acc  = df["correct"].mean()
        out["avg"] = (round(float(avg_acc), 6), count)
        return out
