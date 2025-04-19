import os
import pandas as pd
import numpy as np


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


def categorize_images(df, class_name, class_to_idx, classes):
    class_idx = class_to_idx[class_name]
    prob_cols = classes
    probs = df[prob_cols].values

    correct = df[(df["label_id"] == class_idx) & (df["prediction"] == class_idx)]

    top2 = np.argsort(-probs, axis=1)[:, :2]
    second_best_mask = (
        (df["label_id"] == class_idx)
        & (top2[:, 1] == class_idx)
        & (df["prediction"] != class_idx)
    )
    second_best = df[second_best_mask]

    worst_mask = (df["label_id"] == class_idx) & (np.argmin(probs, axis=1) == class_idx)
    worst = df[worst_mask]

    def get_first_row(df_sub):
        return df_sub.iloc[0] if not df_sub.empty else None

    return get_first_row(correct), get_first_row(second_best), get_first_row(worst)
