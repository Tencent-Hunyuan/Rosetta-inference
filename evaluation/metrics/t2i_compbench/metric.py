from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode

from .blip_vqa import blip_vqa
from ..base_metric import BaseMetric
from rosetta.utils import safe_file


class T2ICompBenchMetric(BaseMetric):
    def __init__(self, vqa_model_path, dataset_name="T2ICompBench", max_size=1024):
        super().__init__()
        self.vqa_model_path = Path(vqa_model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if not self.vqa_model_path.exists():
            raise FileNotFoundError(f"{self.vqa_model_path} not found.")
        self._model = None
        self.transform = T.Compose(
            [
                T.Resize((480, 480), interpolation=InterpolationMode.BICUBIC, antialias=True),
                T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ]
        )
        self.type2int = {
            "color": 0,
            "shape": 1,
            "texture": 2,
        }
        self.dataset_name = dataset_name
        self.max_size = max_size
        self.results = []
        # {} for timestamp
        self.save_file_template = "t2i_compbench_{}.csv"

    def load_model(self, logger=None):
        self._model = blip_vqa(pretrained=self.vqa_model_path, image_size=480, vit="base").to(self.device)
        self._model.eval()
        if logger is not None:
            logger.info("T2ICompBenchMetric: BLIP-VQA model loaded.")

    def release_model(self):
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self.load_model()
        return self._model

    @staticmethod
    def question_loader(questions):
        max_question_len = max(len(q) for q in questions)
        for i in range(max_question_len):
            yield [q[i] if i < len(q) else "" for q in questions]

    @torch.no_grad()
    def process(self, images, questions, dataset_types, prompt, ids, **kwargs):
        """
             que1  que2  que3      score
        im1   a     d     g    -->  adg
        im2   b     e          -->  be
        im3   c     f          -->  cf

        avg_score = (adg + be + cf) / 3

        Args:
            images (torch.Tensor): batch of image tensors with shape (B, 3, H, W)
            questions (list of list of str): list of questions
            dataset_types (list of str): list of dataset type
            **kwargs:
        """
        transformed_image = self.transform(images)

        max_question_len = max(len(q) for q in questions)
        scores = np.ones((len(images), max_question_len), dtype=np.float32)
        for qi, ques in enumerate(self.question_loader(questions)):
            probs = self.model(transformed_image, ques).detach().cpu().numpy()
            for i, (que, prob) in enumerate(zip(ques, probs)):
                if que:
                    scores[i, qi] = prob
        prod_scores = np.prod(scores, axis=1)
        score_type = np.array([self.type2int[dt] for dt in dataset_types])
        self.results.append((np.stack((prod_scores, score_type), axis=1), prompt, ids, dataset_types))

    def compute_metrics(self, results, save_file=None):
        prompts, ids, predictions, dataset_types = [], [], [], []
        for pred, prompt, id_, dataset_type in results:
            predictions.append(pred)
            prompts.extend(prompt)
            ids.extend(id_)
            dataset_types.extend(dataset_type)
        predictions = np.concatenate(predictions, axis=0)
        if save_file is not None:
            df = pd.DataFrame(predictions, columns=["score", "type"])
            df["prompt"] = prompts
            df["index"] = ids
            df["dataset_type"] = dataset_types
            df = df[["index", "dataset_type", "prompt", "score"]]
            df_sorted = df.sort_values(by=["index"])
            df_sorted.to_csv(safe_file(save_file), index=False)
        out_dict = {}

        pred0 = predictions[predictions[:, 1] == 0, 0]
        count0 = pred0.shape[0]
        if count0:
            print("T2ICompBenchMetric: predictions(color).shape is", count0)
            out_dict["color"] = (float(pred0.mean()), count0)

        pred1 = predictions[predictions[:, 1] == 1, 0]
        count1 = pred1.shape[0]
        if count1:
            print("T2ICompBenchMetric: predictions(shape).shape is", count1)
            out_dict["shape"] = (float(pred1.mean()), count1)

        pred2 = predictions[predictions[:, 1] == 2, 0]
        count2 = pred2.shape[0]
        if count2:
            print("T2ICompBenchMetric: predictions(texture).shape is", count2)
            out_dict["texture"] = (float(pred2.mean()), count2)

        out_dict["avg"] = (
            float((pred0.sum() + pred1.sum() + pred2.sum()) / (count0 + count1 + count2)),
            count0 + count1 + count2
        )

        return out_dict
