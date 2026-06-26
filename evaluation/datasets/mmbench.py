import os
import base64
import random
import string

import torch
import pandas as pd
from PIL import Image
from io import BytesIO


def cn_string(s):
    import re
    if re.search(u'[\u4e00-\u9fff]', s):
        return True
    return False


def decode_base64_to_image_file(base64_str, path):
    img_data = base64.b64decode(base64_str)
    img = Image.open(BytesIO(img_data)).convert("RGB")
    img.save(path)


class MMBenchDataset(torch.utils.data.Dataset):
    """Load MMBench dataset directly from TSV file, no VLMEvalKit required."""

    def __init__(self, LMUDataRoot, dataset_name, target_size=None, pad_color=(127, 127, 127)):
        self.dataset_name = dataset_name
        self.img_root = os.path.join(LMUDataRoot, "images", dataset_name)
        os.makedirs(self.img_root, exist_ok=True)

        tsv_path = os.path.join(LMUDataRoot, f"{dataset_name}.tsv")
        assert os.path.exists(tsv_path), (
            f"MMBench TSV file not found: {tsv_path}. "
            f"Please download it to {LMUDataRoot}."
        )
        self.data = pd.read_csv(tsv_path, sep='\t')
        self.data['index'] = self.data['index'].astype(str)

        # Resolve image references: some rows may point to another row's index
        if 'image' in self.data.columns:
            image_map = {row['index']: str(row['image']) for _, row in self.data.iterrows()}
            resolved = {}
            for k, v in image_map.items():
                if len(v) <= 64:   # short string → it's a reference index, not base64
                    assert v in image_map and len(image_map[v]) > 64, \
                        f"Image reference {v!r} for index {k!r} cannot be resolved."
                    resolved[k] = image_map[v]
                else:
                    resolved[k] = v
            self.data['image'] = [resolved[idx] for idx in self.data['index']]

        self.metric_input_key = "answers"
        self.run_fn_kwargs = {"max_new_tokens": 512}

    def __len__(self):
        return len(self.data)

    def _dump_image(self, line):
        """Decode base64 image and save to disk; return file path."""
        idx = line['index']
        img_path = os.path.join(self.img_root, f"{idx}.jpg")
        if not os.path.exists(img_path):
            decode_base64_to_image_file(line['image'], img_path)
        return img_path

    def _build_prompt(self, line):
        question = line['question'] if not pd.isna(line.get('question', float('nan'))) else ""
        hint = line.get('hint', None)
        if hint is not None and not pd.isna(hint):
            question = str(hint) + "\n" + question

        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        for key, item in options.items():
            question += f"\n{key}. {item}"

        if options:
            question += (
                "\n请直接回答选项字母。"
                if cn_string(question)
                else "\nAnswer with the option's letter from the given choices directly."
            )
        else:
            question += (
                "\n请直接回答问题。"
                if cn_string(question)
                else "\nAnswer the question directly."
            )
        return question

    def __getitem__(self, idx):
        line = self.data.iloc[idx]
        question_id = int(line['index']) if str(line['index']).isdigit() else idx
        image_path = self._dump_image(line)
        question = self._build_prompt(line)

        return {
            'id': question_id,
            # `input` is a multimodal message content list: image path + text.
            # parse_dataset in csv_dataset.py will wrap this into message_list.
            'input': [{"type": "image", "image": image_path}, {"type": "text", "text": question}],
            'seed': random.randint(0, 1_000_000),
            'lines': line,          # renamed to "lines" for metric compatibility
        }
