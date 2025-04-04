import pytorch_lightning as pl
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from CNNNetwork import CNNNetwork
from config import Config
import torch.nn.functional as F
import torch
import numpy as np


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
        # Reset counters
        self.train_correct = 0
        self.train_total = 0
        self.train_loss = []

    def on_validation_epoch_end(self):
        # Calculate epoch accuracy
        epoch_acc = self.val_correct / self.val_total
        self.log("val_acc_epoch", epoch_acc)
        if len(self.val_loss) > 0:
            self.log("val_loss_epoch", torch.cat(self.val_loss).mean())
        # Reset counters
        self.val_correct = 0
        self.val_total = 0
        self.val_loss = []

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config.LR)
