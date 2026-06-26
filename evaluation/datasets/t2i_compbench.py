import pandas as pd
from torch.utils.data import Dataset


class T2ICompBenchDataset(Dataset):
    def __init__(self, csvs, debug=False):
        super().__init__()
        if isinstance(csvs, str):
            csvs = [csvs]
        data = []
        for csv in csvs:
            df = pd.read_csv(csv, header=0)
            data.append(df)
        self.data = pd.concat(data, ignore_index=True)
        self.data["index"] = self.data.index
        if debug:
            # Randomly sample 96 samples for debugging
            self.data = self.data.sample(n=96, random_state=0).sort_values("index")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data.iloc[index]
        ret = {
            "id": item["index"],
            "type": "prompt",
            "input": item["prompt"],
            "seed": item["seed"],
            "questions": item["questions"].split("##"),
            "dataset_type": item["dataset_type"],
        }
        return ret

    @staticmethod
    def collate_fn(batch):
        batch_size = len(batch)

        ids = []
        types = []
        inputs = []
        seeds = []
        questions = []
        dataset_types = []

        for i in range(batch_size):
            ids.append(batch[i]["id"])
            types.append(batch[i]["type"])
            inputs.append(batch[i]["input"])
            seeds.append(batch[i]["seed"])
            questions.append(batch[i]["questions"])
            dataset_types.append(batch[i]["dataset_type"])

        ret = {
            "ids": ids,
            "type": types,
            "input": inputs,
            "seeds": seeds,
            "questions": questions,
            "dataset_types": dataset_types,
        }

        return ret
