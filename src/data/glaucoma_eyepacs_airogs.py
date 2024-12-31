from typing import Any, Dict, Optional, Tuple
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
import glob
import os
from torchvision.transforms import transforms
import torchvision
import pandas as pd 
import random


class BP_Eye_Dataset(Dataset):
    def __init__(self, image_paths, classes, my_transforms):
        self.image_paths = image_paths
        self.classes = classes
        self.my_transforms = my_transforms

        self.idx_to_class = {i:j for i, j in enumerate(self.classes)}
        self.class_to_idx = {value:key for key,value in self.idx_to_class.items()}
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx] 
        label = image_filepath.split('/')[-2]
        image = torchvision.io.read_image(image_filepath)
        
        image = self.my_transforms(image)
        label_final = self.class_to_idx[label]

        return image, label_final


my_transforms = transforms.Compose([
    transforms.ToPILImage(), 
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), # convert 0-255 to 0-1 and from np to tensors
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    transforms.Resize(size=(512,512),antialias=True)
])
    

class GlaucomaEyepacsModule(LightningDataModule):
    def __init__(self, batch_size: int, num_workers: int, pin_memory: bool, data_dir:str, classes:list) -> None:
        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=True)
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.batch_size_per_device = batch_size
        self.data_dir = data_dir
        self.classes = classes

    def setup(self, stage: Optional[str] = None) -> None:
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size}).")
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            train_data_paths = sorted(glob.glob(self.data_dir+"train/RG/*.jpg")) + sorted(glob.glob(self.data_dir+"train/NRG/*.jpg"))[:5000] # from train path
            val_data_paths = sorted(glob.glob(self.data_dir+"val/RG/*.jpg")) + sorted(glob.glob(self.data_dir+"val/NRG/*.jpg"))[:700] # from validation path
            test_data_paths = sorted(glob.glob(self.data_dir+"test/RG/*.jpg")) + sorted(glob.glob(self.data_dir+"test/NRG/*.jpg"))[:700] # from test path

            # doing shuffling in the val data because, at a batch there can be all negative case, & even if model is ideal, the metrics like precision, recall is 0%
            # shuffling it now, and will not shuffle it later on during data loading......
            random.shuffle(train_data_paths)
            random.shuffle(val_data_paths)
            random.shuffle(test_data_paths)

            self.data_train = BP_Eye_Dataset(train_data_paths, classes=self.classes, my_transforms=my_transforms)
            self.data_val = BP_Eye_Dataset(val_data_paths, classes=self.classes, my_transforms=my_transforms)
            self.data_test = BP_Eye_Dataset(test_data_paths, classes=self.classes, my_transforms=my_transforms)

    def train_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,)

if __name__ == "__main__":
    _ = GlaucomaEyepacsModule()