import pytorch_lightning as pl
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from CNNNetwork import CNNNetwork
from config import Config
import torch.nn.functional as F
import torch


class LightningModule(pl.LightningModule):
    def __init__(
        self,
        config: Config,
    ):
        super(LightningModule, self).__init__()
        self.config = config
        ## Build the network
        self.CNNmodel = CNNNetwork(config=self.config)
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.CNNmodel(x)

    def training_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(
        self,
        batch,
    ):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config.LR)
