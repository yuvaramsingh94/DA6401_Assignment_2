import os
import pandas as pd


def dir_to_df(PATH: os.path) -> pd.DataFrame:
    data_dict = {"image_path": [], "label": []}
    for label in os.listdir(PATH):
        if os.path.isdir(os.path.join(PATH, label)):
            image_path = [
                os.path.join(os.path.join(PATH, label), i)
                for i in os.listdir(os.path.join(PATH, label))
                if "jpg" in i
            ]
            label_list = [label] * len(image_path)
            data_dict["image_path"].extend(image_path)
            data_dict["label"].extend(label_list)

    data_df = pd.DataFrame.from_dict(data_dict)
    return data_df
