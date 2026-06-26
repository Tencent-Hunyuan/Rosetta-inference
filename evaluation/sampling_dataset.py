import json
from copy import deepcopy
from pathlib import Path
from typing import Optional, Callable

import pandas as pd
from torch.utils.data import Dataset
from loguru import logger

from evaluation.datasets import DATASETS, load_dataset
from evaluation.constants import TESTSET_TEMPLATE, ASSETS_BASE
from rosetta.utils import except_collate_fn


def decode_csv_file(csv, error_message=''):
    if csv is None:
        raise ValueError(f"File not found: {csv}. {error_message}")
    if Path(csv).suffix == "":
        # When src_file has no suffix, we treat it as a stem and fill in the template.
        csv = Path(TESTSET_TEMPLATE.format(csv))
    else:
        csv = Path(csv)
    if not csv.exists():
        raise FileNotFoundError(f"File not found: {csv}. {error_message}")
    return csv


class MessageListDataset(Dataset):
    def __init__(self, testset: str, sample_save_base, tokenizer, skip_existed=False,
                 prompt_fn=None):
        self.tokenizer = tokenizer
        # Define save directory and load existing results if any
        self.save_dir = self.prepare_save_directory(testset, sample_save_base)

        # Define prompt_fn for custom manipulation of the prompt online.
        self.prompt_fn: Optional[Callable[[str, pd.Series], dict]] = prompt_fn
        if self.prompt_fn is None:
            self.prompt_fn = lambda prompt, _: {"role": "user", "content": prompt}

        (
            self.testset_type,
            self.testset,
            self.task_kwargs
        ) = self.parse_testset(testset)
        self.name_mapper = lambda x: self.task_kwargs[x] if x in self.task_kwargs else x

        self.collate_except_keys = []
        if self.testset_type == "csv":
            self.total_input_dict = self.parse_csv_dataset()
            if skip_existed:
                finished_files = list(self.save_dir.glob("results/results_*.csv"))
                if len(finished_files) > 0:
                    finished_indices = set()
                    for file in finished_files:
                        df = pd.read_csv(file, header=0)
                        finished_indices.update(df["index"].tolist())
                    self.total_input_dict = [
                        item for item in self.total_input_dict if item["index"] not in finished_indices
                    ]
                    logger.info(
                        f"Skipped {len(finished_indices)} finished samples, {len(self.total_input_dict)} remaining."
                    )
        elif self.testset_type == "dataset":
            self.total_input_dict = self.parse_dataset()

    @staticmethod
    def prepare_save_directory(testset, sample_save_base):
        testset_renamed, *extra = testset.split("@@")
        if len(extra) > 0:
            testset_renamed += "__" + "_".join([part.split("=")[1] for part in extra])
        sample_save_base = Path(sample_save_base)
        save_base = sample_save_base / Path(testset_renamed).stem
        return save_base.resolve().absolute()

    def parse_testset(self, testset):
        kwargs = {}
        if "@@" in testset:
            testset, *extra = testset.split("@@")
            for part in extra:
                key, value = part.split("=")
                kwargs[key] = value

        if testset in DATASETS and kwargs.get("metric"):
            self.dataset = load_dataset(testset, tokenizer=self.tokenizer)
            testset_type = "dataset"

        else:
            if Path(testset).exists():
                self.testset_file = testset
            else:
                self.testset_file = decode_csv_file(testset)
            testset_type = "csv"

        return testset_type, testset, kwargs

    @staticmethod
    def format_file_path(file_path: Optional[str]):
        if file_path is None:
            return None
        assert isinstance(file_path, str), f"file_path must be str, but got {type(file_path)}"

        file_path = file_path.strip()
        if file_path == "" or file_path.startswith("/") or file_path.startswith("http"):
            return file_path

        # If relative path, Prepend the ASSETS_BASE path
        file_path = Path(ASSETS_BASE) / file_path
        assert file_path.exists(), f"{file_path} does not exist"
        return str(file_path)

    def parse_csv_dataset(self):
        df = pd.read_csv(self.testset_file)
        assert "index" in df.columns, "CSV dataset must contain 'index' column."
        assert "seed" in df.columns, "CSV dataset must contain 'seed' column."

        if (message_col := self.name_mapper("message_list")) in df.columns:
            # OpenAI format message list
            df[message_col] = df[message_col].apply(json.loads)

        elif (prompt_col := self.name_mapper("prompt")) in df.columns and \
            ((src_col := self.name_mapper("src_img_path")) in df.columns or (count_col := self.name_mapper("count")) in df.columns):

            src_col = self.name_mapper("src_img_path")
            count_col = self.name_mapper("count")
            if count_col not in df.columns:
                df["count"] = [1] * len(df)
                count_col = "count"

            df["message_list"] = df.apply(
                lambda row:
                    [{
                        "role": "user",
                        "content": [{
                            "type": "image",
                            "image": self.format_file_path(
                                row[src_col if src_col in df.columns else self.name_mapper("src_img_path_1")]
                            )
                        }]
                    }] +
                    [
                        {
                            "role": "user",
                            "content": [{
                                "type": "image",
                                "image": self.format_file_path(row[self.name_mapper(f"src_img_path_{i+1}")])
                            }]
                        } for i in range(1, row[count_col])
                    ] +
                    [
                        self.prompt_fn(row[prompt_col], row),
                    ],
                axis=1,
            )

        elif (prompt_col := self.name_mapper("prompt")) in df.columns:
            df["message_list"] = df.apply(lambda row: [self.prompt_fn(row[prompt_col], row)], axis=1)

        else:
            raise NotImplementedError(
                f"[MessageListDataset] Unsupported CSV dataset format with columns: {df.columns}."
            )

        self.collate_except_keys = list(set(df.columns) - {"index", "seed", "prompt"})

        return df.to_dict(orient="records")

    def parse_dataset(self):
        data = []
        for i in range(len(self.dataset)):
            src_item = self.dataset[i]
            # Support both "input" (text-only datasets like MMLU) and "input" as a
            # multimodal content list (MMMU/MMBench with PIL image + text dict).
            # Fall back to "prompt" key if "input" is absent.
            input_data = src_item.get("input", src_item.get("prompt", ""))
            # Extract plain text for the "prompt" field (used for display / saving).
            if isinstance(input_data, list):
                prompt_text = next(
                    (item.get("text", "") for item in input_data if item.get("type") == "text"),
                    str(input_data),
                )
            else:
                prompt_text = input_data

            data_item = dict(
                index=src_item["id"],
                seed=src_item["seed"],
                prompt=prompt_text,
                message_list=[
                    {"role": "user", "content": input_data}
                ],
            )
            assert "message_list" not in src_item, \
                "Key conflict: dataset item already contains 'message_list' key."
            remain_keys = [k for k in src_item.keys() if k not in ["id", "seed", "input", "prompt"]]
            for k in remain_keys:
                if k in ["is_dummy"]:
                    # The `is_dummy` in dataset item is for backward compatibility, we skip it here
                    # We will add `is_dummy` flag in __getitem__.
                    continue
                data_item[k] = src_item[k]
            data.append(data_item)

        self.collate_except_keys = list(set(data[0].keys()) - {"index", "seed", "prompt"})

        return data

    def __len__(self):
        return len(self.total_input_dict)

    def __getitem__(self, index):
        is_dummy = index // len(self) > 0
        index = index % len(self)

        data = deepcopy(self.total_input_dict[index])
        data["is_dummy"] = is_dummy
        return data

    def collate_fn(self, batch):
        result = except_collate_fn(batch, except_keys=self.collate_except_keys)
        # Convert list of dicts/Series to pd.DataFrame for metric compatibility
        # (MMMUMetric and MMBenchMetric both expect a pd.DataFrame named "lines").
        if "lines" in result and isinstance(result["lines"], list):
            result["lines"] = pd.DataFrame(result["lines"])
        return result
