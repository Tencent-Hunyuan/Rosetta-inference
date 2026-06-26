import spacy
import random
from pathlib import Path
import pandas as pd


class T2ICompBenchDataset(object):
    def __init__(self, root):
        self.root = Path(root)
        self.nlp = spacy.load("en_core_web_sm")

    def text2questions(self, dataset_name, dataset_type, save_path=None):
        dataset_file = self.root / f"{dataset_name}.txt"
        data_lines = dataset_file.read_text().splitlines()
        qid = 0
        data = []
        for prompt in data_lines:
            prompt = prompt.strip()
            # Skip empty lines
            if not prompt:
                continue

            questions = []
            doc = self.nlp(prompt)
            for chunk in doc.noun_chunks:
                if chunk.text not in ["top", "the side", "the left", "the right"]:
                    questions.append(f"{chunk.text}?")
            data.append(
                {
                    "index": qid,
                    "seed": random.randint(0, 1000000),
                    "prompt": prompt,
                    "questions": "##".join(questions),
                    "dataset": dataset_name,
                    "dataset_type": dataset_type,
                }
            )
            qid += 1

        df = pd.DataFrame(data)
        print(df)

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(save_path, index=False)


if __name__ == "__main__":
    # ds = T2ICompBenchDataset("data/t2i_compbench")
    # ds.text2questions("color_val", "color", "data/t2i_compbench/color_val.csv")
    # ds.text2questions("shape_val", "shape", "data/t2i_compbench/shape_val.csv")
    # ds.text2questions("texture_val", "texture", "data/t2i_compbench/texture_val.csv")
    ds = T2ICompBenchDataset("__data/t2i_compbench")
    ds.text2questions("test", "color", "__data/t2i_compbench/test.csv")
