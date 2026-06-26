from pathlib import Path
import json

from PIL import Image
import torch
from torch.utils.data import Dataset


# COCO2014 validation set
class COCOValDataset(Dataset):
    def __init__(self, path, transform):
        super().__init__()
        self.path = Path(path)
        self.image_files = list(self.path.iterdir())
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        with open(self.image_files[index], "rb") as f:
            image = Image.open(f)
            image = image.convert("RGB")
        return {"image": self.transform(image), "image_id": self.image_files[index].name[:-4]}

    @staticmethod
    def collate_fn(batch):
        batch_size = len(batch)

        images = []
        image_ids = []

        for i in range(batch_size):
            images.append(batch[i]["image"])
            image_ids.append(batch[i]["image_id"])

        return {
            "images": torch.stack(images, dim=0),
            "image_ids": image_ids,
        }


# samples randomly sampled from COCO2014 validation set
# COCO30K or COCO6K
class COCODataset(Dataset):
    def __init__(self, path, debug=False):
        super().__init__()
        self.path = Path(path)
        self.captions = json.load(open(self.path, "r"))["annotations"]
        if debug:
            self.captions = self.captions[:1024]

    def __len__(self):
        # return 120 # for test
        return len(self.captions)

    def __getitem__(self, index):
        return {
            "id": self.captions[index]["id"],
            # different captions may correspond to the same image_id
            "image_id": self.captions[index]["image_id"],
            "type": "prompt",
            "input": self.captions[index]["caption"],
            "seed": self.captions[index]["seed"],
        }

    @staticmethod
    def collate_fn(batch):
        batch_size = len(batch)

        ids = []
        image_ids = []
        types = []
        inputs = []
        seeds = []

        for i in range(batch_size):
            ids.append(batch[i]["id"])
            image_ids.append(batch[i]["image_id"])
            types.append(batch[i]["type"])
            inputs.append(batch[i]["input"])
            seeds.append(batch[i]["seed"])

        return {
            "ids": ids,
            "image_ids": image_ids,
            "type": types,
            "input": inputs,
            "seeds": seeds,
        }
