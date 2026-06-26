import json
import random
import string

import torch
import pandas as pd
from PIL import Image


class AI2DDataset(torch.utils.data.Dataset):
    """AI2D (AI2 Diagrams) TEST dataset.

    Science diagram multiple-choice QA.
    Loads from a local directory saved with datasets.save_to_disk()
    (downloaded via scripts/download/download_eval_data.sh).

    Dataset columns (lmms-lab/ai2d):
      image (PIL), question (str), options (list[str]), answer (str: "0"/"1"/"2"/"3")
    """

    def __init__(self, data_path: str, split: str = "test"):
        """
        Args:
            data_path: local directory saved with save_to_disk(), or HF repo id.
            split: ignored when loading from save_to_disk (already a single split).
        """
        try:
            # Try loading a save_to_disk directory (Arrow format)
            from datasets import load_from_disk
            self.dataset = load_from_disk(data_path)
        except Exception:
            # Fall back to HuggingFace hub / parquet format
            from datasets import load_dataset
            self.dataset = load_dataset(data_path, split=split)

        self.metric_input_key = "answers"
        self.run_fn_kwargs = {"max_new_tokens": 5}

    def __len__(self):
        return len(self.dataset)

    def _build_prompt(self, sample):
        question = sample["question"]
        options = sample["options"]   # list of strings
        for i, opt in enumerate(options):
            question += f"\n{string.ascii_uppercase[i]}. {opt}"
        question += "\nAnswer with the option's letter from the given choices directly."
        return question

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        pil_image = sample["image"].convert("RGB")
        question = self._build_prompt(sample)

        options = sample["options"]
        # answer is "0"/"1"/"2"/"3" (index), convert to "A"/"B"/"C"/"D"
        answer_raw = str(sample["answer"]).strip()
        if answer_raw.isdigit():
            answer_letter = string.ascii_uppercase[int(answer_raw)]
        else:
            # already a letter (some dataset variants use letters directly)
            answer_letter = answer_raw.upper()

        line = {
            "id":       idx,
            "question": sample["question"],
            "answer":   answer_letter,
            "options":  json.dumps(options),
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
