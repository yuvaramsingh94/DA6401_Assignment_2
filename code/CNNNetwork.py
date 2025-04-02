import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from config import Config


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
            if self.cnn_activation == "relu":
                self.conv_layers.append(nn.ReLU())
            elif self.cnn_activation == "sigmoid":
                self.conv_layers.append(nn.Sigmoid())

            ## Add the Max pooling
            self.conv_layers.append(nn.MaxPool2d(kernel_size=2))

        ## Work with the dense layer
        ## TODO Have to find a way to adjust the in_features
        self.dense = nn.LazyLinear(out_features=self.num_dense_neurons)
        self.output_layer = nn.Linear(
            in_features=self.num_dense_neurons, out_features=self.num_classes
        )

    def forward(self, x):

        for layer in self.conv_layers:
            x = layer(x)
        x = torch.flatten(x, 1)

        x = self.dense(x)
        x = self.output_layer(x)

        return x
