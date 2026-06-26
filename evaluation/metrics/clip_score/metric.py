import numpy as np
import torch
from torchvision import transforms
import torch.nn.functional as F
import clip

from ..base_metric import BaseMetric


@torch.no_grad()
def calculate_clip_score(prompts, images, clip_model, device):
    texts = clip.tokenize(prompts, truncate=True).to(device=device)

    image_features = clip_model.encode_image(images)
    text_features = clip_model.encode_text(texts)

    scores = F.cosine_similarity(image_features, text_features)
    return scores


class CLIPScoreMetric(BaseMetric):
    def __init__(self, clip_model_path, dataset_name="COCO", max_size=256):
        super().__init__()
        self.clip_model_path = clip_model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._clip_model = None
        self._clip_image_preprocess = None
        self._transform = None
        self.dataset_name = dataset_name
        self.max_size = max_size
        self.results = []

    def load_model(self, logger=None):
        self._clip_model, self._clip_image_preprocess = clip.load(name=self.clip_model_path, device=self.device)
        # simulate clip_image_preprocess on Tensor to speedup
        # https://github.com/openai/CLIP/blob/dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1/clip/clip.py#L79
        self._transform = transforms.Compose(
            [
                transforms.Resize(
                    self._clip_model.visual.input_resolution,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                transforms.CenterCrop(self._clip_model.visual.input_resolution),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ]
        )
        if logger is not None:
            logger.info("CLIPScoreMetric: CLIP model loaded.")

    def release_model(self):
        self._clip_model = None
        self._clip_image_preprocess = None
        self._transform = None

    @property
    def clip_model(self):
        if self._clip_model is None:
            self.load_model()
        return self._clip_model

    @property
    def clip_image_preprocess(self):
        if self._clip_image_preprocess is None:
            self.load_model()
        return self._clip_image_preprocess

    @property
    def transform(self):
        if self._transform is None:
            self.load_model()
        return self._transform

    # b x c x h x w
    @torch.no_grad()
    def process(self, images, prompt, **kwargs):
        clip_scores = calculate_clip_score(prompt, self.transform(images), self.clip_model, self.device)
        self.results.append(clip_scores.cpu().numpy())

    def compute_metrics(self, results):
        predictions = np.concatenate(results, axis=0)
        count = predictions.shape[0]
        print("CLIPScoreMetric: predictions.shape is", predictions.shape)
        # print(predictions)
        return float(predictions.mean()), int(count)

    # calculate clip score directly, such as for rerank
    @torch.no_grad()
    def calculate_clip_score(self, prompts, images, **kwargs):
        return calculate_clip_score(prompts, self.transform(images), self.clip_model, self.device)
