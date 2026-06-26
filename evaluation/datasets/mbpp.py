"""MBPP (Mostly Basic Python Problems) benchmark dataset loader.

Generation-based evaluation: model generates Python code, which is executed
against the provided test cases.  Uses the ``sanitized`` split (427 problems).

Data: google-research-datasets/mbpp → saved locally with save_to_disk().
"""
import torch
from torch.utils.data import Dataset
from datasets import load_from_disk

# 3-shot examples taken from the MBPP train split (task_ids 2, 3, 4).
_FEW_SHOT = '''\
You are an expert Python programmer. Complete the Python function based on the description.

Task: Write a function to find the similar elements from the given two tuple lists.
Tests:
assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)
assert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4)
[BEGIN]
def similar_elements(test_tup1, test_tup2):
    res = tuple(set(test_tup1) & set(test_tup2))
    return res
[DONE]

Task: Write a python function to identify non-prime numbers.
Tests:
assert is_not_prime(2) == False
assert is_not_prime(10) == True
assert is_not_prime(35) == True
[BEGIN]
import math
def is_not_prime(n):
    if n < 2:
        return True
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return True
    return False
[DONE]

Task: Write a function to find the largest integers from a given list of numbers using heap queue algorithm.
Tests:
assert heap_queue_largest([25, 35, 22, 85, 14, 65, 75, 25, 58], 3) == [85, 75, 65]
assert heap_queue_largest([25, 35, 22, 85, 14, 65, 75, 22, 58], 1) == [85]
[BEGIN]
import heapq
def heap_queue_largest(nums, n):
    return heapq.nlargest(n, nums)
[DONE]

'''


def _build_prompt(text: str, test_list: list[str]) -> str:
    tests_str = "\n".join(test_list)
    return (
        _FEW_SHOT
        + f"Task: {text}\nTests:\n{tests_str}\n[BEGIN]\n"
    )


class MBPPDataset(Dataset):
    """MBPP sanitized test set, 3-shot, generation-based (code execution).

    Args:
        data_path: directory saved by HF ``save_to_disk()``.
    """

    def __init__(self, data_path: str, **kwargs):
        super().__init__()
        ds = load_from_disk(data_path)
        self.samples = []
        for item in ds:
            # sanitized split uses "prompt", full split uses "text"
            text = item.get("prompt", item.get("text", ""))
            self.samples.append({
                "task_id":   item["task_id"],
                "text":      text,
                "test_list": item["test_list"],
            })

        self.metric_input_key = "answers"
        # 512 tokens is enough for most short Python functions
        self.run_fn_kwargs = {"max_new_tokens": 512, "total_count": len(self.samples)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        is_dummy = idx // len(self) > 0
        idx = idx % len(self)

        s = self.samples[idx]
        prompt = _build_prompt(s["text"], s["test_list"])

        return {
            "id":        idx,
            "type":      "prompt",
            "input":     prompt,
            "seed":      42,
            # Pass test_list as JSON string so it survives collation
            "labels":    "\n".join(s["test_list"]),
            "subjects":  str(s["task_id"]),
            "is_dummy":  is_dummy,
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
