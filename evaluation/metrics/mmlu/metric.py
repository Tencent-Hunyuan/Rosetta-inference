import torch
import numpy as np
import pandas as pd
import re
from transformers import AutoTokenizer

from ..base_metric import BaseMetric
from rosetta.tokenizer import TOKENIZER_PATH


class MMLUMetric(BaseMetric):
    def __init__(self, dataset_name="mmlu_bench", tokenizer=None, reports_by_subject=False, device=None,
                 prefix_space=False, **kwargs):
        super().__init__()
        self.dataset_name = dataset_name
        self.prefix_space = prefix_space    # 'A' and ' A' are different tokens in HYTokenizer.
        
        assert tokenizer is not None, "Please provide a tokenizer for evaluating @mmlu_bench."
        if isinstance(tokenizer, str) and tokenizer in TOKENIZER_PATH:
            tokenizer = TOKENIZER_PATH[tokenizer]
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)
        else:
            self.tokenizer = tokenizer

        self.report_by_subject = reports_by_subject

        if device is None:
            if torch.distributed.is_initialized():
                device = torch.distributed.get_rank() % 8
            else:
                device = 0
            self.device = f"cuda:{device}" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.results = []

    def load_model(self, logger):
        self.logger = logger
        pass

    def process_logits(self, pred_logits, labels, subjects, **kwargs):
        """Process prompts and compute predictions.

        Args:
            pred_logits: List of torch.Tensor each with shape (1, vocab_size) or torch.Tensor with shape (batch_size, vocab_size)
            labels: List of ground truth labels.
            subjects: List of subject strings.
        """
        for pred_logit, label, subject in zip(pred_logits, labels, subjects):
            # pred_logit: (1, vocab_size)
            try:
                logits = pred_logit.flatten().cpu().float().numpy()

                # Compute probabilities for each choice
                probs = torch.nn.functional.softmax(
                    torch.tensor([
                        logits[self.tokenizer(" A" if self.prefix_space else "A").input_ids[0]],
                        logits[self.tokenizer(" B" if self.prefix_space else "B").input_ids[0]],
                        logits[self.tokenizer(" C" if self.prefix_space else "C").input_ids[0]],
                        logits[self.tokenizer(" D" if self.prefix_space else "D").input_ids[0]],
                    ]),
                    dim=0,
                ).cpu().numpy()

                # Get prediction
                pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs)]
                self.results.append({"subject": subject, "label": label, "pred": pred, "probs": probs})
            except Exception as e:
                self.logger.warning(f"{e}")
                self.results.append({"subject": subject, "label": label, "pred": None, "probs": None})

        return self.results
    
    def process_sequences(self, sequences, labels, subjects, **kwargs):
        """Process sequences and compute predictions.

        Args:
            sequences: List of sequences.
            labels: List of ground truth labels.
            subjects: List of subject strings.
        """
        model_type = kwargs.get("model_type", None)
        if model_type == "hyvlm_instruct":
            # 正则表达式提取 <｜hy_place▁holder▁no▁151｜> 和 <｜hy_place▁holder▁no▁152｜> 之间的答案
            pattern = r'<｜hy_place▁holder▁no▁151｜>(.*?)<｜hy_place▁holder▁no▁152｜>'
        elif model_type == "instruct":
            pattern = ""
        else:
            assert False, "Invalid model type"
        
        for sequence, label, subject in zip(sequences, labels, subjects):
            try:
                # 如果 sequence 是字符串，直接使用；如果是列表，转换为字符串
                sequence_str = sequence if isinstance(sequence, str) else ''.join(sequence)
                
                # 使用正则表达式提取答案, 支持\n和\r;并且移除\n和\r
                sequence_str = sequence_str.replace('\n', '').replace('\r', '')
                match = re.search(pattern, sequence_str, re.DOTALL | re.MULTILINE)
                if match:
                    pred = match.group(1).strip()  # 提取匹配的内容并去除首尾空白
                    probs = 1.0
                else:
                    pred, probs = None, None
                self.results.append({"subject": subject, "label": label, "pred": pred, "probs": probs})
            except Exception as e:
                self.logger.warning(f"{e}")
                self.results.append({"subject": subject, "label": label, "pred": None, "probs": None})
        
        return self.results
    
    def process(self, answers, labels, subjects, **kwargs): 
        """Process answers and compute predictions.

        Args:
            answers: List of strings.
            labels: List of ground truth labels.
            subjects: List of subject strings.
        """
        if answers is None:
            return self.results
        if not kwargs.get("model_type", None):
            return self.process_logits(answers.logits[0], labels, subjects, **kwargs)
        else:
            return self.process_sequences(answers.sequences, labels, subjects, **kwargs)
           

    def compute_metrics(self, results):
        """Compute accuracy and other metrics grouped by subject."""
        df = pd.DataFrame(results)
        out_dict = {}

        # Group by subject and calculate accuracy for each subject
        if self.report_by_subject:
            for subject, group in sorted(df.groupby('subject')):
                subject_score = group['pred'].eq(group['label']).mean()
                out_dict[subject] = (round(float(subject_score), 6), len(group))

        # Calculate overall accuracy
        count = len(df)
        avg_accuracy = df['pred'].eq(df['label']).mean()
        out_dict['avg'] = (
            round(float(avg_accuracy), 6),
            count
        )

        return out_dict


if __name__ == "__main__":
    # Initialize the metric
    tokenizer_name = "Qwen/Qwen3-0.6B"
    metric = MMLUMetric(tokenizer=tokenizer_name)

    # Mock data for testing
    pred_logits = [
        torch.rand(1, 36),  # Random logits for the first example
        torch.rand(1, 36),  # Random logits for the second example
    ]

    # Print the generated logits for verification
    print("Generated pred_logits:")
    for i, logits in enumerate(pred_logits):
        print(f"Example {i + 1}: {logits}")
        
    labels = ["D", "A"]  # Ground truth labels
    subjects = ["math", "science"]  # Subjects for each example

    # Process the predictions
    results = metric.process(pred_logits, labels, subjects)

    # Compute metrics
    metrics = metric.compute_metrics(results)

    # Print results and metrics
    print("Results:", results)
    print("Metrics:", metrics)
