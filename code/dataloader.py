from torchvision.transforms import v2
from torchvision.transforms.v2 import Normalize
from torchvision.transforms import Resize
from torch.utils.data import Dataset
import pandas as pd
from torchvision.io import decode_image


class CustomImageDataset(Dataset):
    def __init__(
        self,
        dataset_df: pd.DataFrame,
        image_normalization: Normalize,
        size: tuple = (256, 256),
        augmentation: bool = False,
    ):
        self.dataset_df = dataset_df
        self.image_normalization = image_normalization
        self.size = size
        self.augmentation = augmentation
        if self.augmentation:
            ## Do augmentation
            self.transform = v2.Compose(
                [
                    Resize(size=size),
                    v2.RandomHorizontalFlip(p=0.5),
                    v2.RandomRotation(
                        degrees=10,
                    ),
                    v2.ColorJitter(),
                ]
            )
        else:
            ## No augmentation
            self.transform = v2.Compose([Resize(size=size)])

    def __len__(self):
        return len(self.dataset_df)

    def __getitem__(self, idx: int):
        img_path = self.dataset_df.iloc[idx]["image_path"]  # .values[0]
        image = decode_image(img_path, mode="RGB")
        image = self.transform(image)
        image = self.image_normalization(image / image.max())

        label = self.dataset_df.iloc[idx]["label_id"]  # .values[0]
        return image, label
