import os
from typing import Optional

import pandas as pd

from ..base_metric import BaseMetric
from rosetta.utils import safe_file


class MMBenchMetric(BaseMetric):
    """Local MMBench metric: computes accuracy directly from TSV answer column.
    No VLMEvalKit / online submission required.
    """

    def __init__(self, LMUDataRoot, dataset_name="MMBench_DEV_EN"):
        super().__init__()
        self.dataset_name = dataset_name
        self.LMUDataRoot = LMUDataRoot
        self.results = []
        self.save_file_template = f"{dataset_name}_{{}}.xlsx"
        # save_file is optional for local eval
        self.compute_metrics_required_args = []

    def load_model(self, logger=None):
        if logger is not None:
            logger.info("MMBenchMetric: loaded (local eval, no VLMEvalKit).")

    def release_model(self):
        pass

    def process(self, answers, lines: pd.DataFrame, ids=None, **kwargs):
        lines = lines.copy(deep=True)
        lines['prediction'] = answers
        self.results.append(lines)

    def compute_metrics(self, results, save_file=None):
        df = pd.concat(results, ignore_index=True)

        if save_file is not None:
            try:
                df.to_excel(safe_file(save_file), index=False)
            except Exception:
                df.to_csv(safe_file(str(save_file).replace('.xlsx', '.csv')), index=False)

        if 'answer' not in df.columns:
            print("[MMBenchMetric] Warning: 'answer' column not found in data, cannot compute accuracy.")
            return 0.0, int(len(df))

        def extract_letter(pred):
            """Extract single letter prediction from model output."""
            if pred is None or (isinstance(pred, float)):
                return ""
            pred = str(pred).strip()
            # If model output a single letter directly
            if len(pred) == 1 and pred.upper() in "ABCDE":
                return pred.upper()
            # Try to find first occurrence of A/B/C/D
            import re
            m = re.search(r'\b([A-E])\b', pred)
            if m:
                return m.group(1).upper()
            # Fallback: first letter in output
            for ch in pred:
                if ch.upper() in "ABCDE":
                    return ch.upper()
            return ""

        df['pred_letter'] = df['prediction'].apply(extract_letter)
        df['correct'] = df['pred_letter'] == df['answer'].astype(str).str.strip().str.upper()

        overall_acc = df['correct'].mean()

        # Per-category breakdown if available
        printable = {"Overall": {"acc": round(float(overall_acc), 4), "num": int(len(df))}}
        if 'category' in df.columns:
            for cat, group in df.groupby('category'):
                printable[str(cat)] = {
                    "acc": round(float(group['correct'].mean()), 4),
                    "num": int(len(group)),
                }

        import json
        print(json.dumps(printable, indent=4, ensure_ascii=False))

        return float(overall_acc), int(len(df))
