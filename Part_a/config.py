import os


class Config:
    def __init__(self):

        self.wandb_project = "lightning_test"
        self.wandb_entity = "scratch_v1"
        self.num_conv_layers = 5
        self.num_filters = 32
        self.filter_size = [5, 5, 3, 3, 3]
        self.num_dense_neurons = 256
        self.cnn_activation = "elu"
        self.augmentation = True
        self.dense_activation = "relu"
        self.num_classes = 10
        self.LR = 0.0000206362263432735
        self.batch_size = 16
        self.epoch = 5
        self.pretrained_bb = False
        self.drop_prob = 0.1
        self.bn = True
        self.dirpath = os.path.join("weights", "part_a")
        self.filename = "test"
