import os
import random
import torch
import pandas as pd
from torch.utils.data import Dataset

from rosetta.tokenizer import load_tokenizer

def _get_eval_instruction_model() -> bool:
    """获取 EVAL_INSTRUCTION_MODEL 环境变量，支持多种布尔值表示"""
    env_value = os.getenv("EVAL_INSTRUCTION_MODEL", "False")
    # 支持多种表示方式：True, true, TRUE, 1, yes, Yes, YES
    if isinstance(env_value, str):
        env_value = env_value.strip().lower()
        return env_value in ("true", "1", "yes", "on")
    return bool(env_value)

# 获取环境变量，EVAL_INSTRUCTION_MODEL
EVAL_INSTRUCTION_MODEL = _get_eval_instruction_model()
# 打印提示信息
print("************************************************")
print(f"EVAL_INSTRUCTION_MODEL: {EVAL_INSTRUCTION_MODEL} (from env: {os.getenv('EVAL_INSTRUCTION_MODEL', 'False')})")
print("When evaluating the Instruct Model, please set the environment variable EVAL_INSTRUCTION_MODEL to True.")
print("************************************************")

CHOICES = ["A", "B", "C", "D"]
random.seed(42)

class MMLUBenchDataset(Dataset):
    """MMLU Dataset with dynamic prompt generation and length control

    Args:
        data_dir (str): Path to data directory containing dev/test subdirectories
        ntrain (int): Number of few-shot examples to use
        tokenizer: tokenizer used for length checking
        max_length (int): Maximum sequence length (default: 2048)
    """

    def __init__(self, data_dir, ntrain=5, max_length=2048, tokenizer=None, tokenizer_class=None, **kwargs):
        super().__init__()

        self.data_dir = data_dir
        self.ntrain = ntrain
        self.max_length = max_length
        self.tokenizer = self._get_tokenizer(tokenizer, tokenizer_class)
        self.subjects = self._get_subjects()
        self.samples = self._load_samples()

        self.metric_input_key = "pred_logits"
        self.run_fn_kwargs = {"return_first_pred_token_logits": True, "total_count": len(self.samples)}

    @staticmethod
    def _get_tokenizer(tokenizer, tokenizer_class):
        assert tokenizer is not None, "`tokenizer` must be specified for MMLU dataset"
        if isinstance(tokenizer, str):
            tokenizer = load_tokenizer(tokenizer, tokenizer_class)
        return tokenizer

    def _get_subjects(self):
        test_dir = os.path.join(self.data_dir, "test")
        return sorted([
            f.split("_test.csv")[0]
            for f in os.listdir(test_dir)
            if f.endswith("_test.csv")
        ])

    def _load_samples(self):
        samples = []
        for subject in self.subjects:
            # Load dev and test data
            dev_df = pd.read_csv(
                os.path.join(self.data_dir, "dev", f"{subject}_dev.csv"),
                header=None
            )
            test_df = pd.read_csv(
                os.path.join(self.data_dir, "test", f"{subject}_test.csv"),
                header=None
            )

            # Create samples for each test example
            for idx in range(len(test_df)):
                seed = random.randint(0, 1000000)
                samples.append({
                    'subject': subject,
                    'dev_df': dev_df,
                    'test_row': test_df.iloc[idx],
                    'seed': seed,
                })
        return samples

    def _generate_prompt(self, dev_df, test_row, subject, k):
        """Construct prompt with few-shot examples"""
        # Format subject header
        prompt = f"The following are multiple choice questions (with answers) about {self._format_subject(subject)}.\n\n"

        # Add few-shot examples
        for i in range(k):
            example = dev_df.iloc[i]
            prompt += self._format_example(example, include_answer=True)

        # Add test question
        prompt += self._format_example(test_row, include_answer=False)
        label = test_row[dev_df.shape[1]-1]  # Last column contains answer

        return prompt, label

    @staticmethod
    def _format_subject(subject):
        return subject.replace("_", " ")

    @staticmethod
    def _format_example(row, include_answer=True):
        """Format single question example"""
        question = row[0]
        choices = [f"{CHOICES[i]}. {row[i+1]}" for i in range(len(row)-2)]
        # prompt = f"{question}\n" + "\n".join(choices) + "\nAnswer:"
        if include_answer or not EVAL_INSTRUCTION_MODEL:
            prompt = f"{question}\n" + "\n".join(choices) + "\nAnswer:"
        else:
            instruction_prompt = "The above are some example questions and correct answers; now you need to correctly answer the following questions:\n"
            output_prompt = "Now, Strictly output ONLY the final answer—providing just the uppercase option letter (e.g., A) for multiple-choice questions or the direct result for others—without any explanations, reasoning, prefixes, or extra punctuation."
            prompt = instruction_prompt + "\n" + f"{question}\n" + "\n".join(choices) + "\n" + output_prompt

        # Add ground truth answer
        if include_answer:
            prompt += f" {row[len(row) - 1]}\n\n"
        return prompt

    def __len__(self):
        return len(self.samples)  # 14042

    def __getitem__(self, idx):
        is_dummy = idx // len(self) > 0
        idx = idx % len(self)

        sample = self.samples[idx]
        subject = sample['subject']
        dev_df = sample['dev_df']
        test_row = sample['test_row']
        seed = sample['seed']

        # Generate initial prompt
        k = min(self.ntrain, len(dev_df))
        prompt, label = self._generate_prompt(dev_df, test_row, subject, k)

        # Check length: truncate if needed
        if self.tokenizer is not None:
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
            while input_ids.shape[1] > self.max_length and k > 0:
                k -= 1
                prompt, label = self._generate_prompt(dev_df, test_row, subject, k)
                input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
            length = input_ids.shape[1]
        else:
            while len(prompt.split()) > self.max_length and k > 0:
                k -= 1
                prompt, label = self._generate_prompt(dev_df, test_row, subject, k)
            length = len(prompt.split())

        return {
            "id": idx,
            "type": "prompt",
            "input": prompt,
            "seed": seed,
            "labels": label,
            "subjects": subject,
            "length": length,
            "is_dummy": is_dummy,
        }

    @staticmethod
    def collate_fn(batch):
        batch_size = len(batch)

        ids = []
        types = []
        inputs = []
        seeds = []
        labels = []
        subjects = []
        lengths = []

        for i in range(batch_size):
            ids.append(batch[i]["id"])
            types.append(batch[i]["type"])
            inputs.append(batch[i]["input"])
            seeds.append(batch[i]["seed"])
            labels.append(batch[i]["label"])
            subjects.append(batch[i]["subject"])
            lengths.append(batch[i]["length"])
        is_dummy = torch.tensor([item["is_dummy"] for item in batch])

        ret = {
            "ids": ids,
            "type": types,
            "input": inputs,
            "seeds": seeds,
            "labels": labels,
            "subjects": subjects,
            "lengths": lengths,
            "is_dummy": is_dummy,
        }

        return ret


if __name__ == "__main__":
    from evaluation.constants import MMLU_BENCH_DATA
    data_dir = MMLU_BENCH_DATA
    ntrain = 5
    dataset = MMLUBenchDataset(data_dir, ntrain)
    print(len(dataset))
    for i in range(10):
        print(f"{dataset[i]}\n")

# PYTHONPATH="." python3 -m evaluation.datasets.mmlu_bench
