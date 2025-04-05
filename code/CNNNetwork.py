import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from config import Config
from torchvision.models import resnet50, ResNet50_Weights

## TODO: add Dropout
## TODO: better preprocessing and model freezing


class CNNNetwork(nn.Module):
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
                )
            )
            for _ in range(self.num_conv_layers - 1):

                self.conv_layers.append(
                    nn.Conv2d(
                        in_channels=self.num_filters,
                        out_channels=self.num_filters,
                        kernel_size=self.filter_size,
                    )
                )
                self.conv_layers.append(self.act_select(act=self.cnn_activation))

                ## Add the Max pooling
                self.conv_layers.append(nn.MaxPool2d(kernel_size=2))

        ## Work with the dense layer
        self.FCN_layers = nn.ModuleList()
        self.FCN_layers.append(nn.LazyLinear(out_features=self.num_dense_neurons))
        self.FCN_layers.append(self.act_select(act=self.dense_activation))
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
        x = torch.flatten(x, 1)

        for layer in self.FCN_layers:
            x = layer(x)

        return x
