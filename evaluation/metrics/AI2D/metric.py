import json
import re

import numpy as np
import pandas as pd

from ..base_metric import BaseMetric


def extract_letter(pred: str) -> str:
    """Extract single choice letter from model output."""
    if pred is None or (isinstance(pred, float) and np.isnan(pred)):
        return ""
    pred = str(pred).strip()
    # Direct single letter
    if len(pred) == 1 and pred.upper() in "ABCDE":
        return pred.upper()
    # First word
    first = pred.split()[0].rstrip(".,!?;:") if pred.split() else ""
    if len(first) == 1 and first.upper() in "ABCDE":
        return first.upper()
    # Regex
    m = re.search(r'\b([A-E])\b', pred)
    if m:
        return m.group(1).upper()
    # Last resort: first letter in ABCDE
    for ch in pred:
        if ch.upper() in "ABCDE":
            return ch.upper()
    return ""


class AI2DMetric(BaseMetric):
    """Accuracy metric for AI2D_TEST multiple-choice benchmark."""

    def __init__(self):
        super().__init__()
        self.results = []
        self.compute_metrics_required_args = []

    def load_model(self, logger=None):
        if logger is not None:
            logger.info("AI2DMetric: loaded.")

    def release_model(self):
        pass

    def process(self, answers, lines: pd.DataFrame, **kwargs):
        lines = lines.copy(deep=True)
        lines["prediction"] = answers
        self.results.append(lines)

    def compute_metrics(self, results, save_file=None):
        df = pd.concat(results, ignore_index=True)

        if save_file is not None:
            df.to_csv(str(save_file).replace(".xlsx", ".csv"), index=False)

        df["pred_letter"] = df["prediction"].apply(extract_letter)
        df["correct"] = df["pred_letter"] == df["answer"].astype(str).str.strip().str.upper()

        overall_acc = df["correct"].mean()
        printable = {
            "Overall": {
                "acc": round(float(overall_acc), 4),
                "num": int(len(df)),
            }
        }

        print(json.dumps(printable, indent=4, ensure_ascii=False))
        return float(overall_acc), int(len(df))
