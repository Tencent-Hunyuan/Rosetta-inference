from pathlib import Path
import numpy as np
from scipy import linalg
import torch
from torch.nn.functional import adaptive_avg_pool2d

from ..base_metric import BaseMetric
from .inception import InceptionV3


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ("fid calculation produces singular product; " "adding %s to diagonal of cov estimates") % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


class FIDMetric(BaseMetric):
    def __init__(
        self,
        dims,
        inception_path,
        img_save_path: str = None,
        target_path: str = None,
        dataset_name="COCO",
        max_size=256,
    ):
        super().__init__()
        self.inception_path = inception_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        self._inception = None
        self.img_save_path = img_save_path
        self.target_path = target_path
        self.dataset_name = dataset_name
        self.max_size = max_size
        self.results = []

    def load_model(self, logger=None):
        self._inception = InceptionV3([self.block_idx], inception_score=False, inception_path=self.inception_path).to(
            self.device
        )
        self._inception.eval()
        if logger is not None:
            logger.info("FIDMetric: Inception model loaded.")

    def release_model(self):
        self._inception = None

    @property
    def inception(self):
        if self._inception is None:
            self.load_model()
        return self._inception

    def save_images(self, images, image_ids):
        print("Saving to target path.")
        for idx, image in enumerate(images):
            sub_idx = 0
            while Path(f"{self.img_save_path}/image_{image_ids[idx]}_{sub_idx}.png").exists():
                sub_idx += 1
            image.save(f"{self.img_save_path}/image_{image_ids[idx]}_{sub_idx}.png")

    # b x c x h x w
    @torch.no_grad()
    def process(self, images, image_ids=None, **kwargs):
        if self.img_save_path is not None:
            assert image_ids is not None, "image_ids must be provided to save images."
            self.save_images(images, image_ids)

        # b x 2048 x 1 x 1
        prediction = self.inception(images)[0]
        if prediction.size(2) != 1 or prediction.size(3) != 1:
            prediction = adaptive_avg_pool2d(prediction, output_size=(1, 1))
        # b x 2048
        prediction = prediction.flatten(1, 3).cpu().numpy()
        self.results.append(prediction)

    def compute_metrics(self, results):
        # n x 2048
        predictions = np.concatenate(results, axis=0)
        count = predictions.shape[0]
        print("FIDMetric: predictions.shape is", predictions.shape)
        mu_prediction, sigma_prediction = np.mean(predictions, axis=0), np.cov(predictions, rowvar=False)
        targets = torch.load(self.target_path, weights_only=False)
        mu_target, sigma_target = targets["mu"], targets["sigma"]
        # for test
        # mu_target = np.random.randn(2048)
        # sigma_target = np.eye(2048)
        fid = calculate_frechet_distance(mu_prediction, sigma_prediction, mu_target, sigma_target)
        return float(fid), int(count)

    def compute_stats(self, results):
        # n x 2048
        predictions = np.concatenate(results, axis=0)
        print("FIDMetric: predictions.shape is", predictions.shape)
        mu, sigma = np.mean(predictions, axis=0), np.cov(predictions, rowvar=False)
        return mu, sigma
