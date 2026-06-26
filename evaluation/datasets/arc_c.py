"""ARC-Challenge benchmark dataset loader.

Logit-based 0-shot evaluation (same mechanism as MMLU).
Data: allenai/ai2_arc → ARC-Challenge test split, saved with save_to_disk().
"""
import torch
from torch.utils.data import Dataset
from datasets import load_from_disk

CHOICES = ["A", "B", "C", "D"]


def _normalize_key(key: str) -> str:
    """Normalize numeric labels 1/2/3/4 → A/B/C/D."""
    if key in ("1", "2", "3", "4"):
        return CHOICES[int(key) - 1]
    return key.upper()


class ARCCDataset(Dataset):
    """ARC-Challenge test set, 0-shot, logit-based (like MMLU).

    Args:
        data_path: directory saved by HF ``save_to_disk()``.
        tokenizer: tokenizer instance or name string (required for logit eval).
        tokenizer_class: optional tokenizer class name.
    """

    def __init__(self, data_path: str, tokenizer=None, tokenizer_class=None, **kwargs):
        super().__init__()
        from rosetta.tokenizer import load_tokenizer

        if tokenizer is not None and isinstance(tokenizer, str):
            tokenizer = load_tokenizer(tokenizer, tokenizer_class)
        self.tokenizer = tokenizer

        ds = load_from_disk(data_path)
        self.samples = list(ds)

        self.metric_input_key = "pred_logits"
        self.run_fn_kwargs = {
            "return_first_pred_token_logits": True,
            "total_count": len(self.samples),
        }

    def _format_prompt(self, item) -> tuple[str, str]:
        question = item["question"]
        texts  = item["choices"]["text"]
        labels = item["choices"]["label"]

        prompt = f"{question}\n"
        for text, label in zip(texts, labels):
            prompt += f"{_normalize_key(label)}. {text}\n"
        prompt += "Answer:"
        answer = _normalize_key(item["answerKey"])
        return prompt, answer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        is_dummy = idx // len(self) > 0
        idx = idx % len(self)

        prompt, label = self._format_prompt(self.samples[idx])
        return {
            "id":       idx,
            "type":     "prompt",
            "input":    prompt,
            "seed":     42,
            "labels":   label,
            "subjects": "arc_challenge",
            "is_dummy": is_dummy,
        }

    @staticmethod
    def collate_fn(batch):
        return {
            "ids":      [item["id"]       for item in batch],
            "type":     [item["type"]     for item in batch],
            "input":    [item["input"]    for item in batch],
            "seeds":    [item["seed"]     for item in batch],
            "labels":   [item["labels"]   for item in batch],
            "subjects": [item["subjects"] for item in batch],
            "is_dummy": torch.tensor([item["is_dummy"] for item in batch]),
        }
