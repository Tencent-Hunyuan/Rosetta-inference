import json
import re
import string

import numpy as np
import pandas as pd

from ..base_metric import BaseMetric


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation and whitespace."""
    if not text:
        return ""
    text = str(text).lower().strip()
    text = text.rstrip(string.punctuation + " ")
    return text.strip()


def extract_letter(pred: str) -> str:
    """Try to extract a single option letter (A-E) from model prediction."""
    if pred is None or (isinstance(pred, float) and np.isnan(pred)):
        return ""
    pred = str(pred).strip()
    if len(pred) == 1 and pred.upper() in "ABCDE":
        return pred.upper()
    first = pred.split()[0].rstrip(".,!?;:") if pred.split() else ""
    if len(first) == 1 and first.upper() in "ABCDE":
        return first.upper()
    m = re.search(r'\b([A-E])\b', pred)
    if m:
        return m.group(1).upper()
    return ""


def is_correct(pred_text: str, gt: str) -> bool:
    """Unified comparison handling letter-choice, yes/no, and text answers.

    Strategy:
      - If GT is a single option letter (A-E): extract letter from prediction.
      - Otherwise: compare normalized texts (exact match, or GT is prefix of pred).
    """
    gt = str(gt).strip() if gt is not None else ""
    pred_text = str(pred_text).strip() if pred_text else ""

    # MC letter answer
    if len(gt) == 1 and gt.upper() in "ABCDE":
        return extract_letter(pred_text) == gt.upper()

    # Text / yes-no / number answer
    gt_norm   = _normalize(gt)
    pred_norm = _normalize(pred_text)
    if not gt_norm:
        return False
    # Exact match or prediction starts with the GT token
    return pred_norm == gt_norm or pred_norm.startswith(gt_norm)


class RealWorldQAMetric(BaseMetric):
    """Accuracy metric for RealWorldQA (mixed MC + text answers)."""

    def __init__(self):
        super().__init__()
        self.results = []
        self.compute_metrics_required_args = []

    def load_model(self, logger=None):
        if logger is not None:
            logger.info("RealWorldQAMetric: loaded.")

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

        df["correct"] = df.apply(
            lambda row: is_correct(row["prediction"], row["answer"]), axis=1
        )

        # Breakdown by answer type
        letter_mask = df["answer"].astype(str).str.strip().str.upper().isin(list("ABCDE"))
        mc_acc   = float(df[letter_mask]["correct"].mean()) if letter_mask.any() else float("nan")
        text_acc = float(df[~letter_mask]["correct"].mean()) if (~letter_mask).any() else float("nan")
        overall  = float(df["correct"].mean())

        printable = {
            "Overall":          {"acc": round(overall,  4), "num": int(len(df))},
            "MC_letter":        {"acc": round(mc_acc,   4), "num": int(letter_mask.sum())},
            "Text/YesNo/Count": {"acc": round(text_acc, 4), "num": int((~letter_mask).sum())},
        }
        print(json.dumps(printable, indent=4, ensure_ascii=False))
        return overall, int(len(df))
