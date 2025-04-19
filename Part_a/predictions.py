from LightningModule import LightningModule
from config import Config
import os
import torch
import matplotlib.pyplot as plt
from torchvision.transforms.v2 import Normalize
from utils import dir_to_df, categorize_images
from dataloader import CustomImageDataset
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import copy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import wandb
from lightning import seed_everything
import argparse

SEED = 5
seed_everything(SEED, workers=True)

parser = argparse.ArgumentParser(description="HP sweep")
parser.add_argument(
    "--kaggle", action="store_true", help="Set this flag to true if its kaggle"
)
parser.add_argument(
    "--colab", action="store_true", help="Set this flag to true if its colab"
)
parser.add_argument("-w", "--wandb_key", type=str, help="wandb key")
args = parser.parse_args()


config = Config()


## Test dataset loader
DATASET_PATH = os.path.join("dataset", "inaturalist_12K")
WT_PATH = os.path.join("weights", "part_a", "40_epoch_run.ckpt")
if args.kaggle:
    DATASET_PATH = os.path.join(
        "/kaggle", "input", "intro-to-dl-a2-d", "inaturalist_12K"
    )
    WT_PATH = "/kaggle/input/dl-as2-wt/40_epoch_run.ckpt"


lit_model = LightningModule.load_from_checkpoint(
    checkpoint_path=WT_PATH,
    config=config,
)
lit_model = lit_model.eval()
TEST_PATH = os.path.join(DATASET_PATH, "val")
wandb.require("core")

if args.kaggle:
    ## Kaggle secret
    from kaggle_secrets import UserSecretsClient

    secret_label = "wandb_api_key"
    wandb_key = UserSecretsClient().get_secret(secret_label)
    wandb.login(key=wandb_key)
elif args.colab:
    ## Kaggle secret
    # from google.colab import userdata

    # secret_label = "wandb_api_key"
    # wandb_key = userdata.get(secret_label)
    wandb.login(key=args.wandb_key)
else:
    wandb.login()


test_df = dir_to_df(TEST_PATH)
class_name_list = [
    "Reptilia",
    "Animalia",
    "Arachnida",
    "Amphibia",
    "Aves",
    "Mollusca",
    "Fungi",
    "Insecta",
    "Plantae",
    "Mammalia",
]
class_mapping_dict = {j: i for i, j in enumerate(class_name_list)}

test_df["label_id"] = test_df["label"].map(class_mapping_dict)

print(test_df.groupby(["label_id", "label"]).count())
image_normalization = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
test_dataset = CustomImageDataset(
    dataset_df=test_df,
    image_normalization=image_normalization,
    size=(256, 256),
    augmentation=False,
)

wandb.init(
    project=config.wandb_project,
    name="predictions_v1",
    config=config,
)

prob_list = []
pred_list = []
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

for i in tqdm(range(len(test_df))):
    imgs, _ = test_dataset.__getitem__(i)
    logits = lit_model(torch.unsqueeze(imgs, 0).to(device))
    prob = F.softmax(logits, dim=1)  # .cpu().detach().numpy()
    preds = torch.argmax(prob, dim=1).cpu().detach().numpy()
    prob = prob.cpu().detach().numpy()
    pred_list.append(preds)
    prob_list.append(prob)

dummy_df = copy.copy(test_df)
dummy_df[list(class_mapping_dict.keys())] = np.concatenate(prob_list)
dummy_df["prediction"] = np.concatenate(pred_list)
test_accuracy = accuracy_score(dummy_df["label_id"], dummy_df["prediction"])
print("Test accuracy", test_accuracy)

if not os.path.exists(
    os.path.join(
        "weights",
        "part_a",
    )
):
    os.makedirs(
        os.path.join(
            "weights",
            "part_a",
        )
    )

## Save the predictions
dummy_df.to_csv(os.path.join("weights", "part_a", "dummy_v1.csv"), index=False)


classes = list(class_mapping_dict.keys())

class_to_idx = {cls: i for i, cls in enumerate(classes)}
idx_to_class = {i: cls for i, cls in enumerate(classes)}


# Plotting
fig, axes = plt.subplots(10, 3, figsize=(15, 30))
fig.subplots_adjust(hspace=0.5, wspace=0.5)

for i, cls in enumerate(classes):
    correct, second_best, worst = categorize_images(
        dummy_df, cls, class_to_idx, classes
    )

    def display_image(ax, row, title):
        if row is not None and os.path.exists(row["image_path"]):
            img = Image.open(row["image_path"])
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(title)
        else:
            ax.text(0.5, 0.5, "Image Not Found", fontsize=12, ha="center")
            ax.axis("off")

    display_image(axes[i, 0], correct, f"Correct: {cls}")

    if second_best is not None:
        pred_class_second_best = idx_to_class[second_best["prediction"]]
        display_image(
            axes[i, 1], second_best, f"2nd Best: {cls}\nPred: {pred_class_second_best}"
        )
    else:
        display_image(axes[i, 1], None, f"2nd Best: {cls}\nPred: N/A")

    if worst is not None:
        pred_class_worst = idx_to_class[worst["prediction"]]
        display_image(axes[i, 2], worst, f"Worst: {cls}\nPred: {pred_class_worst}")
    else:
        display_image(axes[i, 2], None, f"Worst: {cls}\nPred: N/A")

plt.tight_layout()
wandb.log({"Prediction samples": fig})
wandb.log({"Test accuracy": test_accuracy})
