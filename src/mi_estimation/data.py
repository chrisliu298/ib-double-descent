import os

import scipy.io as sio
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, TensorDataset

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


class SZTDataModule(BaseDataModule):
    def __init__(self, config):
        super().__init__(config)
        self.dataset_path = config.dataset_path

    def setup(self, stage=None):
        assert os.path.isfile(self.dataset_path), f"{self.dataset_path} does not exist."
        data = sio.loadmat(self.dataset_path)
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
