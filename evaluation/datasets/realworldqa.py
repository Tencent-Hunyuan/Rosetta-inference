import random
import re
import string

import torch
import pandas as pd


class RealWorldQADataset(torch.utils.data.Dataset):
    """RealWorldQA benchmark (xai-org/RealworldQA).

    Practical spatial-understanding multiple-choice questions on real-world images.
    Loaded from a directory saved with datasets.save_to_disk().

    HF columns: image (PIL), question (str, may contain embedded options), answer (str letter).
    """

    def __init__(self, data_path: str, split: str = "test"):
        try:
            from datasets import load_from_disk
            self.dataset = load_from_disk(data_path)
        except Exception:
            from datasets import load_dataset
            self.dataset = load_dataset(data_path, split=split)

        self.metric_input_key = "answers"
        self.run_fn_kwargs = {"max_new_tokens": 10}  # text answers can be a few tokens

    def __len__(self):
        return len(self.dataset)

    def _build_prompt(self, sample):
        question = sample["question"]
        # If options are already embedded in the question text, keep as-is.
        # Otherwise, if a 'choices' column exists, append them.
        is_mc = any(f"{ch}." in question for ch in "ABCDE")

        if "choices" in sample and sample["choices"]:
            choices = sample["choices"]
            if not any(f"{string.ascii_uppercase[i]}." in question
                       for i in range(len(choices))):
                for i, opt in enumerate(choices):
                    question += f"\n{string.ascii_uppercase[i]}. {opt}"
                is_mc = True

        # Only add letter instruction for MC questions.
        # Non-MC questions already include their own instruction (e.g. "Please answer yes or no.").
        if is_mc:
            question += "\nAnswer with the option's letter from the given choices directly."
        return question

    def _get_answer_letter(self, sample):
        """Return the raw GT answer as-is; the metric handles comparison."""
        return str(sample.get("answer", "")).strip()

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        pil_image = sample["image"].convert("RGB")
        question = self._build_prompt(sample)
        answer_letter = self._get_answer_letter(sample)

        line = {
            "id":     idx,
            "answer": answer_letter,
        }

        return {
            "id":    idx,
            "input": [
                {"type": "image", "image": pil_image},
                {"type": "text",  "text":  question},
            ],
            "seed":  random.randint(0, 1_000_000),
            "lines": line,
        }
