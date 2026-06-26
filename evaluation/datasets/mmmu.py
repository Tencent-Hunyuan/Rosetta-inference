import re
import random

import torch
import pandas as pd
from torchvision.transforms import transforms
from PIL import Image
from datasets import load_dataset, concatenate_datasets

from rosetta.utils import to_2tuple


CAT_SHORT2LONG = {
    'acc': 'Accounting',
    'agri': 'Agriculture',
    'arch': 'Architecture_and_Engineering',
    'art': 'Art',
    'art_theory': 'Art_Theory',
    'bas_med': 'Basic_Medical_Science',
    'bio': 'Biology',
    'chem': 'Chemistry',
    'cli_med': 'Clinical_Medicine',
    'cs': 'Computer_Science',
    'design': 'Design',
    'diag_med': 'Diagnostics_and_Laboratory_Medicine',
    'econ': 'Economics',
    'elec': 'Electronics',
    'ep': 'Energy_and_Power',
    'fin': 'Finance',
    'geo': 'Geography',
    'his': 'History',
    'liter': 'Literature',
    'manage': 'Manage',
    'mark': 'Marketing',
    'mate': 'Materials',
    'math': 'Math',
    'mech': 'Mechanical_Engineering',
    'music': 'Music',
    'phar': 'Pharmacy',
    'phys': 'Physics',
    'psy': 'Psychology',
    'pub_health': 'Public_Health',
    'socio': 'Sociology'
}

DOMAIN_CAT2SUB_CAT = {
  'Art and Design': ['Art', 'Art_Theory', 'Design', 'Music'],
  'Business': ['Accounting', 'Economics', 'Finance', 'Manage','Marketing'],
  'Science': ['Biology', 'Chemistry', 'Geography', 'Math', 'Physics',],
  'Health and Medicine': ['Basic_Medical_Science', 'Clinical_Medicine', 'Diagnostics_and_Laboratory_Medicine', 'Pharmacy', 'Public_Health'],
  'Humanities and Social Science': ['History', 'Literature', 'Sociology', 'Psychology'],
  'Tech and Engineering': ['Agriculture', 'Architecture_and_Engineering', 'Computer_Science', 'Electronics', 'Energy_and_Power', 'Materials', 'Mechanical_Engineering'],
}

CONFIG = dict(
    task_instructions="",
    multi_choice_example_format="""{}
{}
Answer with the option's letter from the given choices directly.""",
    short_ans_example_format="""{}
Answer the question using a single word or phrase.""",
)


def parse_img_path(text):
    matches = re.findall("<img='(.*?)'>", text)
    return matches


def process_single_sample(data):
    question = data['question']
    o_imgs_paths = []
    for option in data['options']:
        current_o_imgs_paths = parse_img_path(option)
        for img_path in current_o_imgs_paths:
            o_imgs_paths.append(img_path)

    if len(o_imgs_paths) > 1:  # multiple images in options, used for random selection
        return {'id': data['id'], 'question': question, 'options': data['options'], 'answer': data['answer'],
                'image': None, 'question_type': data['question_type']}
    else:
        return {'id': data['id'], 'question': question, 'options': data['options'], 'answer': data['answer'],
                'image': data['image_1'], 'question_type': data['question_type']}


def construct_prompt(sample, config):
    question = sample['question']
    options = eval(sample['options'])
    example = ""
    if sample['question_type'] == 'multiple-choice':
        start_chr = 'A'
        prediction_range = []
        index2ans = {}
        for option in options:
            prediction_range.append(start_chr)
            example += f"({start_chr}) {option}\n"
            index2ans[start_chr] = option
            start_chr = chr(ord(start_chr) + 1)
        empty_prompt_sample_structure = config['multi_choice_example_format']
        empty_prompt = empty_prompt_sample_structure.format(question, example)
        res_dict = {}
        res_dict['index2ans'] = index2ans
        res_dict['correct_choice'] = sample['answer']
        res_dict['all_choices'] = prediction_range
        res_dict['empty_prompt'] = empty_prompt
        if config['task_instructions']:
            res_dict['final_input_prompt'] = config['task_instructions'].strip() + '\n\n' + empty_prompt
        else:
            res_dict['final_input_prompt'] = empty_prompt

        res_dict['gt_content'] = options[ord(sample['answer'].upper()) - ord('A')]
    else:
        empty_prompt_sample_structure = config['short_ans_example_format']
        empty_prompt = empty_prompt_sample_structure.format(question)
        res_dict = {}
        res_dict['empty_prompt'] = empty_prompt
        if config['task_instructions']:
            res_dict['final_input_prompt'] = config['task_instructions'].strip() + '\n\n' + empty_prompt
        else:
            res_dict['final_input_prompt'] = empty_prompt
        res_dict['gt_content'] = sample['answer']

    res_dict.update(sample)
    return res_dict


class MMMUDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_name, data_path, split, target_size, pad_color=(127, 127, 127)):
        self.data_path = data_path
        self.split = split
        self.dataset_name = dataset_name
        self.target_size = to_2tuple(target_size)
        self.pad_color = pad_color

        # run for each subject
        sub_dataset_list = []
        for subject in CAT_SHORT2LONG.values():
            sub_dataset = load_dataset(data_path, subject, split=split)
            sub_dataset_list.append(sub_dataset)
        # merge all dataset
        self.dataset = concatenate_datasets(sub_dataset_list)

        self.pil_image_to_tensor = transforms.Compose(
            [
                transforms.ToTensor(),  # convert to tensor and normalize to [0, 1]
                transforms.Normalize([0.5], [0.5]),  # transform to [-1, 1]
            ]
        )
        self.metric_input_key = "answers"
        self.run_fn_kwargs = {"max_new_tokens": 10}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        sample = process_single_sample(sample)
        sample = construct_prompt(sample, CONFIG)
        # {
        #     "index2ans": {"A": "$6", "B": "$7", "C": "$8", "D": "$9"},
        #     "correct_choice": "B",
        #     "all_choices": ["A", "B", "C", "D"],
        #     "empty_prompt": "<image 1> Baxter Company has a relevant range of production between 15,000 and 30,000 units. The following cost data represents average variable costs per unit for 25,000 units of production. If 30,000 units are produced, what are the per unit manufacturing overhead costs incurred?\n(A) $6\n(B) $7\n(C) $8\n(D) $9\n\nAnswer with the option's letter from the given choices directly.",
        #     "final_input_prompt": "<image 1> Baxter Company has a relevant range of production between 15,000 and 30,000 units. The following cost data represents average variable costs per unit for 25,000 units of production. If 30,000 units are produced, what are the per unit manufacturing overhead costs incurred?\n(A) $6\n(B) $7\n(C) $8\n(D) $9\n\nAnswer with the option's letter from the given choices directly.",
        #     "gt_content": "$7",
        #     "id": "validation_Accounting_1",
        #     "question": "<image 1> Baxter Company has a relevant range of production between 15,000 and 30,000 units. The following cost data represents average variable costs per unit for 25,000 units of production. If 30,000 units are produced, what are the per unit manufacturing overhead costs incurred?",
        #     "options": "['$6', '$7', '$8', '$9']",
        #     "answer": "B",
        #     "question_type": "multiple-choice"
        # }

        pil_image = sample.pop('image').convert('RGB')

        # Strip "<image N>" placeholders from the prompt; the image is passed
        # separately as a message-list content item.
        import re
        prompt_text = re.sub(r'<image\s*\d+>\s*', '', sample['final_input_prompt']).strip()

        line = {
            'id': sample['id'],
            'question_type': sample['question_type'],
            'answer': sample['answer'],
            'category': '_'.join(sample['id'].split('_')[1:-1]),
            **(
                {
                    'all_choices': sample['all_choices'],
                    'index2ans': sample['index2ans'],
                }
                if 'all_choices' in sample else {}
            ),
            'final_input_prompt': sample['final_input_prompt'],
        }

        return {
            'id': sample['id'],     # str, e.g., "validation_Accounting_1"
            # `input` is a multimodal message content list: image (PIL) + text.
            # parse_dataset in csv_dataset.py will wrap this into message_list.
            'input': [{"type": "image", "image": pil_image}, {"type": "text", "text": prompt_text}],
            'seed': random.randint(0, 1_000_000),
            'lines': line,          # renamed to "lines" for metric compatibility
        }

    @staticmethod
    def collate_fn(batch):
        batch_size = len(batch)

        ids = []
        seeds = []
        prompts = []
        lines = []

        images = torch.stack([sample["image"] for sample in batch], 0)
        for i in range(batch_size):
            ids.append(batch[i]["id"])
            seeds.append(batch[i]["seed"])
            prompts.append(batch[i]["prompt"])
            lines.append(batch[i]["line"])

        ret = {
            "ids": ids,
            "image": images,
            "prompt": prompts,
            "seeds": seeds,
            "lines": pd.DataFrame(lines),
        }

        return ret
