from config import Config
from CNNNetwork import CNNNetwork
from LightningModule import LightningModule
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
import torch
import wandb
from lightning.pytorch.loggers import WandbLogger
from torchvision.transforms.v2 import Normalize
from torchvision.transforms import Resize
from torch.utils.data import Dataset
import pandas as pd
from torchvision.io import read_image, decode_image
import os
from sklearn.model_selection import StratifiedShuffleSplit
from lightning.pytorch import Trainer, seed_everything
from utils import dir_to_df
from dataloader import CustomImageDataset
import argparse

SEED = 5
seed_everything(SEED, workers=True)

## Get the Required keys from secrets if its in kaggle
parser = argparse.ArgumentParser(description="HP sweep")
parser.add_argument(
    "--kaggle", action="store_true", help="Set this flag to true if its kaggle"
)
parser.add_argument(
    "--colab", action="store_true", help="Set this flag to true if its colab"
)
parser.add_argument("-w", "--wandb_key", type=str, help="wandb key")
# Parse the arguments
args = parser.parse_args()


if args.kaggle:
    ## Kaggle secret
    from kaggle_secrets import UserSecretsClient

    secret_label = "wandb_api_key"
    wandb_key = UserSecretsClient().get_secret(secret_label)
    wandb.login(key=wandb_key)


if args.colab:
    ## Kaggle secret
    # from google.colab import userdata

    # secret_label = "wandb_api_key"
    # wandb_key = userdata.get(secret_label)
    wandb.login(key=args.wandb_key)

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


sweep_configuration = {
    "method": "random",
    "metric": {"goal": "maximize", "name": "val_acc_epoch"},
    "parameters": {
        "learning_rate": {"max": 0.0001, "min": 0.00001},
        "CNN_filters": {"values": [32, 64, 128]},
        "CNN_filter_size": {"values": [3, 5]},
        "num_dense_neurons": {"values": [128, 256, 512, 1024]},
        "batch_size": {"values": [16, 32, 64]},
        "augmentation": {"values": [True, False]},
        "basic_CNN": {"values": [False]},
        "pretrained_bb": {"values": [True]},
        "cnn_activation": {
            "values": [
                "relu",
                "elu",
                "silu",
                "mish",
                "gelu",
            ]
        },
        "dense_activation": {
            "values": [
                "relu",
                "elu",
                "silu",
                "mish",
                "gelu",
            ]
        },
    },
    #    "early_terminate": {"type": "hyperband", "min_iter": 3, "eta": 3},
}

image_normalization = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def main():
    """
    The main function that has all the code to run a training.
    This will be used by the sweep agent to run multiple hyperparamter
    tuning.
    """
    config = Config()
    if args.kaggle:

        wandb.init(
            # Set the project where this run will be logged
            project=config.wandb_project,
            # Track hyperparameters and run metadata
            # config=config,
        )
    else:

        wandb.init(
            # Set the project where this run will be logged
            project=config.wandb_project,
            # Track hyperparameters and run metadata
            # config=config,
        )

    wandb.run.name = f"basic_CNN_CF_{wandb.config.CNN_filters}_D_{wandb.config.num_dense_neurons}_A_{wandb.config.augmentation}"
    ## Update the config dict with the hpt from sweep
    config.LR = wandb.config.learning_rate
    config.batch_size = wandb.config.batch_size
    config.num_filters = wandb.config.CNN_filters
    config.filter_size = wandb.config.CNN_filter_size
    config.num_dense_neurons = wandb.config.num_dense_neurons
    config.augmentation = wandb.config.augmentation
    config.cnn_activation = wandb.config.cnn_activation
    config.dense_activation = wandb.config.dense_activation
    config.pretrained_bb = wandb.config.pretrained_bb

    train_dataset = CustomImageDataset(
        dataset_df=train_set,
        image_normalization=image_normalization,
        size=(256, 256),
        augmentation=config.augmentation,
    )
    val_dataset = CustomImageDataset(
        dataset_df=val_set,
        image_normalization=image_normalization,
        size=(256, 256),
        augmentation=False,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=2,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=2,
    )
    lit_model = LightningModule(config=config)
    wandb_logger = WandbLogger(
        project=config.wandb_project,
        name=config.wandb_entity,
        log_model=False,
        config=config,
    )
    trainer = pl.Trainer(
        max_epochs=config.epoch,
        accelerator="auto",
        log_every_n_steps=None,
        logger=wandb_logger,
    )  # Added accelerator gpu, can be cpu also, devices set to 1

    trainer.fit(
        lit_model,
        train_loader,
        val_loader,
    )


config = Config()
## initialize the HPT
sweep_id = wandb.sweep(sweep=sweep_configuration, project=config.wandb_project)

wandb.agent(sweep_id, function=main, count=50)
