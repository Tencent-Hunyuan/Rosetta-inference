import json
import re

import numpy as np
import pandas as pd

from ..base_metric import BaseMetric


def extract_yes_no(pred: str) -> str:
    """Extract yes/no from model output."""
    if pred is None or (isinstance(pred, float) and np.isnan(pred)):
        return "unknown"
    pred = str(pred).strip().lower()
    # Direct match
    if pred in ("yes", "no"):
        return pred
    # First word
    first_word = pred.split()[0].rstrip(".,!?;:") if pred.split() else ""
    if first_word in ("yes", "no"):
        return first_word
    # Search in text
    if re.search(r'\byes\b', pred):
        return "yes"
    if re.search(r'\bno\b', pred):
        return "no"
    return "unknown"


class POPEMetric(BaseMetric):
    """Computes POPE accuracy / precision / recall / F1 per subset and overall."""

    def __init__(self):
        super().__init__()
        self.results = []
        self.compute_metrics_required_args = []

    def load_model(self, logger=None):
        if logger is not None:
            logger.info("POPEMetric: loaded.")

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

        df["pred_yn"] = df["prediction"].apply(extract_yes_no)
        df["correct"] = df["pred_yn"] == df["answer"].astype(str).str.lower().str.strip()

        def metrics_for(sub_df):
            tp = ((sub_df["pred_yn"] == "yes") & (sub_df["answer"] == "yes")).sum()
            fp = ((sub_df["pred_yn"] == "yes") & (sub_df["answer"] == "no")).sum()
            fn = ((sub_df["pred_yn"] == "no")  & (sub_df["answer"] == "yes")).sum()
            tn = ((sub_df["pred_yn"] == "no")  & (sub_df["answer"] == "no")).sum()
            acc       = (tp + tn) / max(len(sub_df), 1)
            precision = tp / max(tp + fp, 1)
            recall    = tp / max(tp + fn, 1)
            f1        = 2 * precision * recall / max(precision + recall, 1e-9)
            yes_rate  = (sub_df["pred_yn"] == "yes").mean()
            return dict(acc=round(float(acc), 4),
                        precision=round(float(precision), 4),
                        recall=round(float(recall), 4),
                        f1=round(float(f1), 4),
                        yes_rate=round(float(yes_rate), 4),
                        num=int(len(sub_df)))

        printable = {}
        if "subset" in df.columns:
            for subset, group in df.groupby("subset"):
                printable[str(subset)] = metrics_for(group)
        printable["Overall"] = metrics_for(df)

        print(json.dumps(printable, indent=4, ensure_ascii=False))

        # Return overall F1 as the primary score (standard for POPE)
        return float(printable["Overall"]["f1"]), int(printable["Overall"]["num"])
