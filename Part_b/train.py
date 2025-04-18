from config import Config
from CNNNetwork import CNNNetwork
from LightningModule import LightningModule
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
import torch
import wandb
from lightning.pytorch.loggers import WandbLogger
import pandas as pd
from torchvision.transforms.v2 import Normalize
import os
from sklearn.model_selection import StratifiedShuffleSplit
from lightning.pytorch import Trainer, seed_everything
from dataloader import CustomImageDataset
from utils import dir_to_df
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

## Dataloader
DATASET_PATH = os.path.join("dataset", "inaturalist_12K")
if args.kaggle:
    DATASET_PATH = os.path.join(
        "/kaggle", "input", "intro-to-dl-a2-d", "inaturalist_12K"
    )
TRAIN_PATH = os.path.join(DATASET_PATH, "train")
TEST_PATH = os.path.join(DATASET_PATH, "val")

data_df = dir_to_df(TRAIN_PATH)
test_df = dir_to_df(TEST_PATH)
class_mapping_dict = {j: i for i, j in enumerate(test_df["label"].unique())}
data_df["label_id"] = data_df["label"].map(class_mapping_dict)
test_df["label_id"] = test_df["label"].map(class_mapping_dict)
## Randomize the dataframe
data_df = data_df.sample(frac=1.0)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)

# Perform the split
for train_idx, val_idx in split.split(data_df, data_df["label_id"]):
    train_set = data_df.iloc[train_idx]
    val_set = data_df.iloc[val_idx]


config = Config()
image_normalization = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_dataset = CustomImageDataset(
    dataset_df=train_set, image_normalization=image_normalization
)
val_dataset = CustomImageDataset(
    dataset_df=val_set, image_normalization=image_normalization
)

train_loader = DataLoader(
    train_dataset,
    batch_size=config.batch_size,
    shuffle=True,
    drop_last=True,
    pin_memory=True,
    # num_workers=2,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=config.batch_size,
    shuffle=False,
    drop_last=False,
    pin_memory=True,
    # num_workers=2,
)

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

lit_model = LightningModule(config=config)
wandb_logger = WandbLogger(
    project=config.wandb_project,
    name=config.wandb_entity,
    log_model="all",
    config=config,
)
trainer = pl.Trainer(
    max_epochs=5,
    accelerator="auto",
    log_every_n_steps=100,
    logger=wandb_logger,
)  # Added accelerator gpu, can be cpu also, devices set to 1

trainer.fit(
    lit_model,
    train_loader,
    val_loader,
)
