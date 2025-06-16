import base64
import os

import datasets
import pandas as pd

from src.logger import logger
from src.utils import assemble_project_path


class GAIADataset:
    def __init__(self, path, name, split):
        self.path = path
        self.name = name
        self.split = split

        path = assemble_project_path(path)
        ds = datasets.load_dataset(path, name, trust_remote_code=True)[split]
        ds = ds.rename_columns(
            {"Question": "question", "Final answer": "true_answer", "Level": "task"}
        )
        ds = ds.map(
            self.preprocess_file_paths,
            load_from_cache_file=False,
            fn_kwargs={"split": split, "path": path},
        )

        data = pd.DataFrame(ds)

        self.data = data

    def preprocess_file_paths(self, row, path, split):
        save_path = assemble_project_path(os.path.join(path, "2023", split))
        os.makedirs(save_path, exist_ok=True)
        if len(row["file_name"]) > 0:
            row["file_name"] = os.path.join(save_path, row["file_name"])
        return row

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data.iloc[index]


class HLEDataset:
    def __init__(self, path, name, split):
        self.path = path
        self.name = name
        self.split = split

        path = assemble_project_path(path)
        ds = datasets.load_dataset(path, trust_remote_code=True)[split]
        ds = ds.rename_columns({"answer": "true_answer", "id": "task_id"})
        ds = ds.map(
            self.preprocess_file_paths,
            load_from_cache_file=False,
            fn_kwargs={"split": split, "path": path},
        )

        data = pd.DataFrame(ds)
        self.data = data

    def preprocess_file_paths(self, row, path, split):
        save_path = assemble_project_path(os.path.join(path, "images", split))
        os.makedirs(save_path, exist_ok=True)

        image_path = ""
        if len(row["image"]) > 0:
            image_string = row["image"]
            task_id = row["task_id"]
            if image_string.startswith("data:image"):
                image_type = image_string.split(";")[0].split("/")[1]
                image_base64 = image_string.split(",")[1]

                image_path = os.path.join(save_path, f"{task_id}.{image_type}")
                with open(image_path, "wb") as f:
                    f.write(base64.b64decode(image_base64))
                logger.info(f"Save image {task_id} to {image_path}")
            else:
                image_path = ""

        row["file_name"] = image_path
        return row

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data.iloc[index]
