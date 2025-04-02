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

SEED = 5
seed_everything(SEED, workers=True)

"""
# Create dummy data
num_samples = 64
input_size = (3, 256, 256)  # Example input size
X = torch.randn(num_samples, *input_size)
y = torch.randint(0, 10, (num_samples,))  # Dummy labels
dataset = TensorDataset(X, y)
train_loader = DataLoader(dataset, batch_size=32)
val_loader = DataLoader(dataset, batch_size=32)
"""


def dir_to_df(PATH: os.path) -> pd.DataFrame:
    data_dict = {"image_path": [], "label": []}
    for label in os.listdir(PATH):
        if os.path.isdir(os.path.join(PATH, label)):
            image_path = [
                os.path.join(os.path.join(PATH, label), i)
                for i in os.listdir(os.path.join(PATH, label))
            ]
            label_list = [label] * len(image_path)
            data_dict["image_path"].extend(image_path)
            data_dict["label"].extend(label_list)

    data_df = pd.DataFrame.from_dict(data_dict)
    return data_df


DATASET_PATH = os.path.join("dataset", "nature_12K", "inaturalist_12K")
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


image_normalization = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


class CustomImageDataset(Dataset):
    def __init__(
        self,
        dataset_df: pd.DataFrame,
        image_normalization: Normalize,
        size: tuple = (256, 256),
    ):
        self.dataset_df = dataset_df
        self.image_normalization = image_normalization
        self.size = size
        self.Resize = Resize(size=size)

    def __len__(self):
        return len(self.dataset_df)

    def __getitem__(self, idx: int):
        img_path = self.dataset_df.iloc[idx]["image_path"]  # .values[0]
        image = decode_image(img_path, mode="RGB")
        image = self.Resize(image)
        image = self.image_normalization(image / image.max())

        label = self.dataset_df.iloc[idx]["label_id"]  # .values[0]
        return image, label


config = Config()
train_dataset = CustomImageDataset(
    dataset_df=train_set, image_normalization=image_normalization
)
val_dataset = CustomImageDataset(
    dataset_df=val_set, image_normalization=image_normalization
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    drop_last=True,
    pin_memory=True,
    #    num_workers=2,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    drop_last=False,
    pin_memory=True,
    #    num_workers=2,
)

# wandb.login()
# wandb.init(
#    config=config,
# )
# Create and train the model
lit_model = LightningModule(config=config)
wandb_logger = WandbLogger(
    project=config.wandb_project,
    name=config.wandb_entity,
    log_model="all",
    config=config,
)
trainer = pl.Trainer(
    max_epochs=5,
    accelerator="gpu",
    log_every_n_steps=100,
    logger=wandb_logger,
)  # Added accelerator gpu, can be cpu also, devices set to 1

trainer.fit(
    lit_model,
    train_loader,
    val_loader,
)
