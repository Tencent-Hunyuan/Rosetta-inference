"""Calculate T2I-CompBench scores after image generation.

Usage:
    PYTHONPATH="." python -m evaluation.metrics.t2i_compbench.offline \
        --image-dir /path/to/samples/t2i_compbench

Paths default to public_assets/evaluation/blip-vqa-capfilt-large and
public_assets/evaluation/T2I-CompBench. Override them with VQA_MODEL_PATH /
T2I_COMPBENCH_PATH environment variables.
"""

import argparse
import json
import os

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from evaluation.constants import T2I_COMPBENCH_PATH, VQA_MODEL_PATH
from evaluation.datasets.t2i_compbench import T2ICompBenchDataset
from evaluation.metrics.t2i_compbench.metric import T2ICompBenchMetric


def parse_args():
    parser = argparse.ArgumentParser(description="Calculate T2I-CompBench scores offline.")
    parser.add_argument(
        "--image-dir", type=str, required=True,
        help="Directory containing the generated images/ sub-folder, e.g. <SAMPLE_OUT>/t2i_compbench."
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()

    image_dir = os.path.abspath(args.image_dir)
    images_sub = os.path.join(image_dir, "images")
    save_dir = os.path.join(image_dir, "metric_results")
    save_path = os.path.join(save_dir, "result.json")
    path_tmpl = os.path.join(images_sub, "{}_0.png")

    print(f"[t2i_compbench] image dir : {image_dir}")

    if os.path.exists(save_path):
        print(f"[t2i_compbench] Already done: {save_path}")
        with open(save_path) as f:
            data = json.load(f)
        for key, value in data.get("results", {}).items():
            print(f"  {key}: {value['score']:.4f}" if isinstance(value, dict) else f"  {key}: {value}")
        return

    if not os.path.isdir(images_sub):
        raise FileNotFoundError(
            f"Images directory not found: {images_sub}\n"
            "Run image generation first."
        )

    print("[t2i_compbench] Loading BLIP-VQA model...")
    metric = T2ICompBenchMetric(vqa_model_path=VQA_MODEL_PATH)
    metric.load_model()

    dataset = T2ICompBenchDataset(csvs=T2I_COMPBENCH_PATH)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=dataset.collate_fn,
        num_workers=args.num_workers,
    )
    print(f"[t2i_compbench] {len(dataset)} samples, {len(loader)} batches")

    missing = 0
    for i, batch in enumerate(loader):
        print(f"  batch {i + 1}/{len(loader)}")
        images = []
        for img_id in batch["ids"]:
            path = path_tmpl.format(img_id)
            try:
                images.append(transforms.ToTensor()(Image.open(path)))
            except FileNotFoundError:
                print(f"  missing: {path}")
                missing += 1
                images.append(torch.zeros(3, 256, 256))
        metric.process(
            images=torch.stack(images).cuda(),
            questions=batch["questions"],
            dataset_types=batch["dataset_types"],
            prompt=batch["input"],
            ids=batch["ids"],
        )

    results = metric.compute_metrics(metric.results)
    print("\n[t2i_compbench] Scores:")
    for key, value in results.items():
        if isinstance(value, tuple):
            print(f"  {key}: {value[0]:.4f}  (n={value[1]})")

    os.makedirs(save_dir, exist_ok=True)
    output = {
        "path_template": path_tmpl,
        "missing_num": missing,
        "results": {
            key: {"score": float(value[0]), "count": int(value[1])} if isinstance(value, tuple) else value
            for key, value in results.items()
        },
    }
    with open(save_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"[t2i_compbench] Saved: {save_path}")


if __name__ == "__main__":
    main()
