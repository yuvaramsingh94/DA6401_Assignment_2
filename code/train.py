from config import Config
from CNNNetwork import CNNNetwork
from LightningModule import LightningModule
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
import torch
import wandb
from lightning.pytorch.loggers import WandbLogger

# Create dummy data
num_samples = 64
input_size = (3, 256, 256)  # Example input size
X = torch.randn(num_samples, *input_size)
y = torch.randint(0, 10, (num_samples,))  # Dummy labels
dataset = TensorDataset(X, y)
train_loader = DataLoader(dataset, batch_size=32)
val_loader = DataLoader(dataset, batch_size=32)
config = Config()
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
    max_epochs=2,
    accelerator="cpu",
    devices=1,
    log_every_n_steps=1,
    logger=wandb_logger,
)  # Added accelerator gpu, can be cpu also, devices set to 1

trainer.fit(
    lit_model,
    train_loader,
    val_loader,
)
