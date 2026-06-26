"""Big Bench Hard (BBH) benchmark dataset loader.

3-shot direct-answer evaluation across all 26 BBH tasks.
Data: downloaded from HuggingFace ``lukaemon/bbh`` and saved locally as
  BBH_PATH/
    {task_name}/
      train.jsonl    (3 few-shot examples)
      test.jsonl     (evaluation examples)
"""
import json
import torch
from pathlib import Path
from torch.utils.data import Dataset


BBH_TASKS = [
    "boolean_expressions",
    "causal_judgement",
    "date_understanding",
    "disambiguation_qa",
    "dyck_languages",
    "formal_fallacies",
    "geometric_shapes",
    "hyperbaton",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
    "logical_deduction_three_objects",
    "movie_recommendation",
    "multistep_arithmetic_two",
    "navigate",
    "object_counting",
    "penguins_in_a_table",
    "reasoning_about_colored_objects",
    "ruin_names",
    "salient_translation_error_detection",
    "snarks",
    "sports_understanding",
    "temporal_sequences",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects",
    "word_sorting",
]


def _load_jsonl(path: Path) -> list[dict]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


class BBHDataset(Dataset):
    """Big Bench Hard test set, 3-shot, generation-based exact-match.

    Args:
        data_path: root directory containing one sub-folder per task.
        n_shot: number of few-shot examples from the train split (default 3).
    """

    def __init__(self, data_path: str, n_shot: int = 3, **kwargs):
        super().__init__()
        root = Path(data_path)
        self.samples = []

        for task in BBH_TASKS:
            task_dir = root / task
            if not task_dir.exists():
                continue

            train_path = task_dir / "train.jsonl"
            test_path  = task_dir / "test.jsonl"
            if not test_path.exists():
                continue

            few_shot_examples = _load_jsonl(train_path)[:n_shot] if train_path.exists() else []
            few_shot_prompt = ""
            for ex in few_shot_examples:
                few_shot_prompt += f"Q: {ex['input']}\nA: {ex['target']}\n\n"

            for ex in _load_jsonl(test_path):
                self.samples.append({
                    "task":          task,
                    "input":         ex["input"],
                    "target":        ex["target"],
                    "few_shot_prompt": few_shot_prompt,
                })

        if not self.samples:
            raise FileNotFoundError(
                f"No BBH data found under {data_path}. "
                "Run the download script first."
            )

        self.metric_input_key = "answers"
        self.run_fn_kwargs = {"max_new_tokens": 128, "total_count": len(self.samples)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        is_dummy = idx // len(self) > 0
        idx = idx % len(self)

        s = self.samples[idx]
        prompt = s["few_shot_prompt"] + f"Q: {s['input']}\nA:"

        return {
            "id":       idx,
            "type":     "prompt",
            "input":    prompt,
            "seed":     42,
            "labels":   s["target"],
            "subjects": s["task"],
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
