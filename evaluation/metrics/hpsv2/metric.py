import numpy as np
import torch
import hpsv2

from ..base_metric import BaseMetric

"""
https://github.com/tgxs002/HPSv2
pip3 install hpsv2
"""


class HPSv2Metric(BaseMetric):
    def __init__(self, hpsv2_model_path, dataset_name="COCO", max_size=256):
        super().__init__()
        version = getattr(hpsv2, "__version__", None)
        if version is None:
            raise NotImplementedError(
                "This version of HPSv2 has been deprecated, please re-install HPSv2 package using commands "
                "in scripts/install_libs.sh."
            )
        self.hpsv2_model_path = hpsv2_model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._hpsv2_model = None
        self._preprocess_val_on_tensor = None
        self._tokenizer = None
        self.dataset_name = dataset_name
        self.max_size = max_size
        self.results = []

    def load_model(self, logger=None):
        self._hpsv2_model, _, _, self._preprocess_val_on_tensor, self._tokenizer = hpsv2.initialize_model(
            self.hpsv2_model_path, self.device
        )
        if logger is not None:
            logger.info("HPSv2Metric: HPSv2 model loaded.")

    def release_model(self):
        self._hpsv2_model = None
        self._preprocess_val_on_tensor = None
        self._tokenizer = None

    @property
    def hpsv2_model(self):
        if self._hpsv2_model is None:
            self.load_model()
        return self._hpsv2_model

    @property
    def preprocess_val_on_tensor(self):
        if self._preprocess_val_on_tensor is None:
            self.load_model()
        return self._preprocess_val_on_tensor

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self.load_model()
        return self._tokenizer

    # b x c x h x w
    @torch.no_grad()
    def process(self, images, prompt, **kwargs):
        hps = hpsv2.batch_score(
            self.hpsv2_model, self.preprocess_val_on_tensor, self.tokenizer, images, prompt, self.device
        )
        self.results.append(hps)

    def compute_metrics(self, results):
        predictions = np.concatenate(results, axis=0)
        count = predictions.shape[0]
        print("HPSv2Metric: predictions.shape is", predictions.shape)
        # print(predictions)
        return float(predictions.mean()), int(count)
