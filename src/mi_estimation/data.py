import os

import numpy as np
import scipy.io as sio
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchvision.datasets import MNIST, FashionMNIST

from utils import train_test_split


class BaseDataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


class ToyDatasetDataModule(BaseDataModule):
    def __init__(self, config):
        super().__init__(config)
        self.root = config.root

    def setup(self, stage=None):
        assert os.path.isfile(self.root), f"{self.root} does not exist."
        data = sio.loadmat(self.root)
        x, y = data["F"], data["y"]
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).squeeze().long()
        x_id = torch.arange(x.shape[0])
        (
            self.x_train,
            self.x_test,
            self.y_train,
            self.y_test,
            self.x_train_id,
            self.x_test_id,
        ) = list(train_test_split(x, y, x_id, total_size=x.shape[0], test_size=0.15))
        self.train_dataset = TensorDataset(self.x_train, self.y_train)
        self.test_dataset = TensorDataset(self.x_test, self.y_test)
