# from pytorch_lightning.utilities.types import OptimizerLRScheduler
from CNNNetwork import CNNNetwork
from config import Config
import torch.nn.functional as F
import torch
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from lightning import LightningModule


class LightningModule(LightningModule):
    def __init__(
        self,
        config: Config,
    ):
        super(LightningModule, self).__init__()
        self.config = config
        ## Build the network
        self.CNNmodel = CNNNetwork(config=self.config)
        self.loss = torch.nn.CrossEntropyLoss()
        # Initialize counters for accuracy calculation
        self.train_correct = 0
        self.train_total = 0
        self.val_correct = 0
        self.val_total = 0

        self.train_loss = []
        self.val_loss = []

    def forward(self, x):
        return self.CNNmodel(x)

    def training_step(self, batch):
        x, y = batch
        logits = self(x)
        prob = F.softmax(logits, dim=1)
        preds = torch.argmax(prob, dim=1)
        correct = (preds == y).sum().item()
        batch_size = y.size(0)

        # Update counters
        self.train_correct += correct
        self.train_total += batch_size

        loss = self.loss(logits, y)
        self.train_loss.append(loss.view(1).cpu())
        # self.log("train_loss", loss)
        return loss

    def validation_step(
        self,
        batch,
    ):
        x, y = batch
        logits = self(x)
        prob = F.softmax(logits, dim=1)
        preds = torch.argmax(prob, dim=1)
        correct = (preds == y).sum().item()
        batch_size = y.size(0)

        # Update counters
        self.val_correct += correct
        self.val_total += batch_size
        loss = self.loss(logits, y)
        self.val_loss.append(loss.view(1).cpu())
        # self.log("val_loss", loss)
        return loss

    def on_train_epoch_end(self):
        # Calculate epoch accuracy
        epoch_acc = self.train_correct / self.train_total
        self.log("train_acc_epoch", epoch_acc)
        if len(self.train_loss) > 0:
            self.log("train_loss_epoch", torch.cat(self.train_loss).mean())
        # Reset lists
        self.train_correct = 0
        self.train_total = 0
        self.train_loss = []

    def on_validation_epoch_end(self):
        # Calculate epoch accuracy
        epoch_acc = self.val_correct / self.val_total
        self.log("val_acc_epoch", epoch_acc)
        if len(self.val_loss) > 0:
            self.log("val_loss_epoch", torch.cat(self.val_loss).mean())
        # Reset lists
        self.val_correct = 0
        self.val_total = 0
        self.val_loss = []

    ## https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.configure_optimizers
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.LR)
        lr_scheduler_config = {
            "scheduler": ReduceLROnPlateau(
                optimizer=optimizer, mode="max", factor=0.1, patience=2
            ),
            "interval": "epoch",
            "frequency": 1,
            "monitor": "val_acc_epoch",
            # If set to `True`, will enforce that the value specified 'monitor'
            # is available when the scheduler is updated, thus stopping
            # training if not found. If set to `False`, it will only produce a warning
            "strict": True,
            "name": "LR_track",
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}
