class Config:
    def __init__(self):

        self.wandb_project = "lightning_test"
        self.wandb_entity = "v1"
        self.num_conv_layers = 5
        self.num_filters = 32
        self.filter_size = 3
        self.num_dense_neurons = 256
        self.cnn_activation = "relu"
        self.dense_activation = "relu"
        self.num_classes = 10
        self.LR = 1e-4
