import os
from typing import Optional

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from pytorch_lightning import LightningDataModule

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ImagenetDataModule(LightningDataModule):
    def __init__(self, data_dir: str, batch_size, image_size: int = 224, num_workers: int = 10):
        super(ImagenetDataModule, self).__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        # transformations
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

        self.val_transform = transforms.Compose([
            transforms.Resize(image_size + 32),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            self.imagenet_train = ImageFolder(os.path.join(self.data_dir, "train"), transform=self.train_transform)
            self.imagenet_val = ImageFolder(os.path.join(self.data_dir, "val"), transform=self.val_transform)
        if stage in (None, "test"):
            raise Exception("Test not available!")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.imagenet_train, batch_size=self.batch_size, shuffle=True, pin_memory=True,
                          num_workers=self.num_workers, persistent_workers=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.imagenet_val, batch_size=self.batch_size, shuffle=False, pin_memory=True,
                          num_workers=self.num_workers, persistent_workers=True)
