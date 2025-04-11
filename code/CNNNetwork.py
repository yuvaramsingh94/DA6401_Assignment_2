import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from config import Config
from torchvision.models import resnet50, ResNet50_Weights
from pytorch_lightning import LightningModule

## TODO: add Dropout
## TODO: better preprocessing and model freezing


class CNNNetwork(LightningModule):
    def __init__(
        self,
        config: Config,
    ):
        super(CNNNetwork, self).__init__()
        self.num_conv_layers = config.num_conv_layers
        self.num_filters = config.num_filters
        self.filter_size = config.filter_size
        self.num_dense_neurons = config.num_dense_neurons
        self.cnn_activation = config.cnn_activation
        self.dense_activation = config.dense_activation
        self.num_classes = config.num_classes
        self.pretrained_bb = config.pretrained_bb
        self.drop_prob = config.drop_prob
        self.bn = config.bn
        if self.pretrained_bb:
            self.bb = resnet50(weights=ResNet50_Weights.DEFAULT)

        else:
            self.conv_layers = nn.ModuleList()
            ## First Conv layer
            self.conv_layers.append(
                nn.Conv2d(
                    in_channels=3,
                    out_channels=self.num_filters,
                    kernel_size=self.filter_size,
                    padding="same",
                )
            )

            self.conv_layers.append(self.act_select(act=self.cnn_activation))
            ## BN
            if self.bn:
                self.conv_layers.append(nn.BatchNorm2d(self.num_filters))
            ## Drop
            # self.conv_layers.append(nn.Dropout(p=self.drop_prob))
            # self.conv_layers.append(nn.MaxPool2d(kernel_size=2))
            for i in range(1, self.num_conv_layers + 1):

                self.conv_layers.append(
                    nn.Conv2d(
                        in_channels=self.num_filters * i,
                        out_channels=self.num_filters * (i + 1),
                        kernel_size=self.filter_size,
                        padding="same",
                    )
                )

                self.conv_layers.append(self.act_select(act=self.cnn_activation))
                ## BN
                if self.bn:
                    self.conv_layers.append(nn.BatchNorm2d(self.num_filters * (i + 1)))

                ## Drop
                # self.conv_layers.append(nn.Dropout(p=self.drop_prob))
                self.conv_layers.append(nn.MaxPool2d(kernel_size=2))

        ## Work with the dense layer
        self.FCN_layers = nn.ModuleList()
        self.FCN_layers.append(nn.LazyLinear(out_features=self.num_dense_neurons))
        # self.FCN_layers.append(nn.BatchNorm1d(self.num_dense_neurons))
        self.FCN_layers.append(self.act_select(act=self.dense_activation))
        self.FCN_layers.append(nn.Dropout(p=self.drop_prob))
        self.FCN_layers.append(
            nn.Linear(in_features=self.num_dense_neurons, out_features=self.num_classes)
        )

    def act_select(self, act: str):
        if act == "relu":
            return nn.ReLU()
        elif act == "elu":
            return nn.ELU()
        elif act == "gelu":
            return nn.GELU()
        elif act == "silu":
            return nn.SiLU()
        elif act == "mish":
            return nn.Mish()

    def forward(self, x):

        if self.pretrained_bb:
            x = self.bb(x)
        else:

            for layer in self.conv_layers:
                x = layer(x)
        # x = torch.flatten(x, 1)
        x = torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)

        for layer in self.FCN_layers:
            x = layer(x)

        return x
