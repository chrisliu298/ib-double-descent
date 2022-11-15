import os

import numpy as np
import scipy.io as sio
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

from utils import standardize, train_test_split


class BaseDataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.batch_size = cfg.batch_size
        self.num_workers = cfg.num_workers

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

    def add_label_noise(self, labels, label_noise, num_classes):
        labels = np.array(labels)
        # indices for noisy labels
        mask = np.random.rand(len(labels)) < label_noise
        # generate random labels
        random_labels = np.random.choice(num_classes, mask.sum())
        labels[mask] = random_labels
        # convert back to original labels format
        labels = torch.tensor([int(x) for x in labels]).long()
        return labels


class SZTDataModule(BaseDataModule):
    """A toy binary classification dataset.

    The dataset is from https://arxiv.org/abs/1703.00810 and is originally included in
    https://github.com/ravidziv/IDNNs.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg

    def setup(self, stage=None):
        assert os.path.isfile(
            f"datasets/{self.cfg.dataset}.mat"
        ), f"datasets/{self.cfg.dataset}.mat not found."
        # Read and split data
        data = sio.loadmat(f"datasets/{self.cfg.dataset}.mat")
        x, y = data["F"], data["y"]
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).squeeze().long()
        # The x_id here is for computing the joint probability p(x, y)
        x_id = torch.arange(x.shape[0])
        (
            self.x_train,
            self.x_test,
            self.y_train,
            self.y_test,
            self.x_train_id,
            self.x_test_id,
        ) = list(train_test_split(x, y, x_id, total_size=x.shape[0], test_size=0.15))
        if self.cfg.label_noise > 0:
            self.y_train = self.add_label_noise(self.y_train, self.cfg.label_noise, 2)
        # Create tensor datasets
        self.train_dataset = TensorDataset(self.x_train, self.y_train)
        self.test_dataset = TensorDataset(self.x_test, self.y_test)


class MNISTDataModule(BaseDataModule):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

    def prepare_data(self):
        # download data
        datasets.MNIST("/tmp/data", train=True, download=True)
        datasets.MNIST("/tmp/data", train=False, download=True)

    def setup(self, stage=None):
        # load data
        self.train_dataset = datasets.MNIST(
            "/tmp/data", train=True, transform=self.transform
        )
        self.test_dataset = datasets.MNIST(
            "/tmp/data", train=False, transform=self.transform
        )
        if self.cfg.label_noise > 0:
            self.train_dataset.targets = self.add_label_noise(
                self.train_dataset.targets, self.cfg.label_noise, 10
            )
        self.x_train, self.y_train = self.train_dataset.data, self.train_dataset.targets
        self.x_test, self.y_test = self.test_dataset.data, self.test_dataset.targets
        self.x_train = standardize(self.x_train, 0.1307, 0.3081).view(-1, 784)
        self.x_test = standardize(self.x_test, 0.1307, 0.3081).view(-1, 784)
        self.x_train_id = torch.arange(self.x_train.shape[0])
        self.x_test_id = torch.arange(self.x_test.shape[0])
