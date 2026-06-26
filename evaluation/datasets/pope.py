import json
import random
import os

import torch
import pandas as pd
from PIL import Image


POPE_SUBSETS = ["adversarial", "popular", "random"]


class POPEDataset(torch.utils.data.Dataset):
    """POPE (Polling-based Object Probing Evaluation) dataset.

    Tests object hallucination with binary yes/no questions.
    Expects data in: {data_root}/coco_pope_{subset}.json
    Images in:       {image_root}/  (COCO val2014)
    """

    def __init__(self, data_root: str, image_root: str, subsets=None):
        """
        Args:
            data_root: directory with coco_pope_adversarial.json etc.
            image_root: directory containing COCO val2014 images.
            subsets: list of subsets to use, defaults to all 3.
        """
        if subsets is None:
            subsets = POPE_SUBSETS

        self.image_root = image_root
        self.data = []

        for subset in subsets:
            json_path = os.path.join(data_root, f"coco_pope_{subset}.json")
            assert os.path.exists(json_path), (
                f"POPE annotation file not found: {json_path}\n"
                f"Please run: bash scripts/download/download_pope.sh"
            )
            with open(json_path) as f:
                for line in f:
                    item = json.loads(line.strip())
                    item["subset"] = subset
                    self.data.append(item)

        self.metric_input_key = "answers"
        self.run_fn_kwargs = {"max_new_tokens": 5}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        image_path = os.path.join(self.image_root, item["image"])
        assert os.path.exists(image_path), f"POPE image not found: {image_path}"

        question = item["text"]   # keep original prompt unchanged (official POPE format)

        line = {
            "question_id": item["question_id"],
            "answer":       item["label"].lower().strip(),   # "yes" or "no"
            "subset":       item["subset"],
            "image":        item["image"],
        }

        return {
            # Use global running idx (not question_id) as unique index.
            # question_id repeats 1-500 across the 3 subsets, which would
            # cause drop_duplicates to incorrectly deduplicate valid samples.
            "id":    idx,
            "input": [
                {"type": "image", "image": image_path},
                {"type": "text",  "text":  question},
            ],
            "seed":  random.randint(0, 1_000_000),
            "lines": line,
        }
