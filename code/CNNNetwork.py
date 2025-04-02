import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


class CNNNetwork(nn.Module):
    def __init__(
        self,
        num_conv_layers: int,
        num_filters: int,
        filter_size: int,
        num_dense_neurons: int,
        cnn_activation: str = "relu",
        dense_activation: str = "relu",
        num_classes: int = 10,
    ):
        super(CNNNetwork, self).__init__()
        self.num_conv_layers = num_conv_layers
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.num_dense_neurons = num_dense_neurons
        self.cnn_activation = cnn_activation
        self.dense_activation = dense_activation
        self.num_classes = num_classes

        self.conv_layers = nn.ModuleList()
        for _ in range(self.num_conv_layers):
            if len(nn.ModuleList()) < 1:  ## First layer
                self.conv_layers.append(
                    nn.Conv2d(
                        in_channels=3, out_channels=num_filters, kernel_size=filter_size
                    )
                )
            else:
                self.conv_layers.append(
                    nn.Conv2d(
                        in_channels=num_filters,
                        out_channels=num_filters,
                        kernel_size=filter_size,
                    )
                )
            if cnn_activation == "relu":
                self.conv_layers.append(nn.ReLU())
            elif cnn_activation == "sigmoid":
                self.conv_layers.append(nn.Sigmoid())

            ## Add the Max pooling
            self.conv_layers.append(nn.MaxPool2d(kernel_size=2))

        ## Work with the dense layer
        ## TODO Have to find a way to adjust the in_features
        self.dense = nn.Linear(
            in_features=num_filters * 12 * 12, out_features=num_dense_neurons
        )
        self.output_layer = nn.Linear(
            in_features=num_dense_neurons, out_features=num_classes
        )

    def forward(self, x):

        for layer in self.conv_layers:
            x = layer(x)
        x = torch.flatten(x, 1)
        x = self.dense(x)
        x = self.output_layer(x)

        return x
