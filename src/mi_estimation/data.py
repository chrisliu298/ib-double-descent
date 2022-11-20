import os

import scipy.io as sio
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

from utils import add_label_noise, make_binary, train_test_split


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
            self.y_train = add_label_noise(self.y_train, self.cfg.label_noise, 2)
        # Create tensor datasets
        self.train_dataset = TensorDataset(self.x_train, self.y_train)
        self.test_dataset = TensorDataset(self.x_test, self.y_test)


class MNISTDataModule(BaseDataModule):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        x_transforms = []
        if cfg.image_size is not None:
            x_transforms.append(transforms.Resize(cfg.image_size))
        x_transforms.extend(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        self.x_transforms = transforms.Compose(x_transforms)
        y_transforms = None
        if cfg.loss == "mse":
            num_classes = int(self.cfg.layer_shapes.split("x")[-1])
            y_transforms = transforms.Compose(
                [
                    lambda y: torch.tensor(y),
                    lambda y: torch.nn.functional.one_hot(y, num_classes),
                    lambda y: y.float(),
                ]
            )
        self.y_transforms = y_transforms

    def prepare_data(self):
        # download data
        datasets.MNIST("/tmp/data", train=True, download=True)
        datasets.MNIST("/tmp/data", train=False, download=True)

    def setup(self, stage=None):
        # load data
        self.train_dataset = datasets.MNIST(
            "/tmp/data",
            train=True,
            transform=self.x_transforms,
            target_transform=self.y_transforms,
        )
        self.test_dataset = datasets.MNIST(
            "/tmp/data",
            train=False,
            transform=self.x_transforms,
            target_transform=self.y_transforms,
        )
        # if (
        #     self.cfg.train_size is not None
        #     and len(self.train_dataset) < self.cfg.train_size
        # ):
        #     self.train_dataset.data, self.train_dataset.targets = sample_data(
        #         self.train_dataset.data, self.train_dataset.targets, self.cfg.train_size
        #     )
        if self.cfg.binary_label:
            binary_labels = torch.tensor([0, 6])
            self.train_dataset.data, self.train_dataset.targets = make_binary(
                self.train_dataset.data, self.train_dataset.targets, binary_labels
            )
            self.test_dataset.data, self.test_dataset.targets = make_binary(
                self.test_dataset.data, self.test_dataset.targets, binary_labels
            )
        if self.cfg.label_noise > 0:
            num_labels = 2 if self.cfg.binary_label else 10
            self.train_dataset.targets = add_label_noise(
                self.train_dataset.targets, self.cfg.label_noise, num_labels
            )
        self.x_train = torch.cat([x[0] for x in self.train_dataset])
        self.x_test = torch.cat([x[0] for x in self.test_dataset])
        self.y_train = torch.stack([x[1] for x in self.train_dataset])
        self.y_test = torch.stack([x[1] for x in self.test_dataset])
        self.x_train_id = torch.arange(self.x_train.shape[0])
        self.x_test_id = torch.arange(self.x_test.shape[0])


class FashionMNISTDataModule(BaseDataModule):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        x_transforms = []
        if cfg.image_size is not None:
            x_transforms.append(transforms.Resize(cfg.image_size))
        x_transforms.extend(
            [transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))]
        )
        self.x_transforms = transforms.Compose(x_transforms)
        y_transforms = None
        if cfg.loss == "mse":
            num_classes = int(self.cfg.layer_shapes.split("x")[-1])
            y_transforms = transforms.Compose(
                [
                    lambda y: torch.tensor(y),
                    lambda y: torch.nn.functional.one_hot(y, num_classes),
                    lambda y: y.float(),
                ]
            )
        self.y_transforms = y_transforms

    def prepare_data(self):
        # download data
        datasets.FashionMNIST("/tmp/data", train=True, download=True)
        datasets.FashionMNIST("/tmp/data", train=False, download=True)

    def setup(self, stage=None):
        # load data
        self.train_dataset = datasets.FashionMNIST(
            "/tmp/data",
            train=True,
            transform=self.x_transforms,
            target_transform=self.y_transforms,
        )
        self.test_dataset = datasets.FashionMNIST(
            "/tmp/data",
            train=False,
            transform=self.x_transforms,
            target_transform=self.y_transforms,
        )
        # if (
        #     self.cfg.train_size is not None
        #     and len(self.train_dataset) < self.cfg.train_size
        # ):
        #     self.train_dataset.data, self.train_dataset.targets = sample_data(
        #         self.train_dataset.data, self.train_dataset.targets, self.cfg.train_size
        #     )
        if self.cfg.binary_label:
            binary_labels = torch.tensor([0, 6])
            self.train_dataset.data, self.train_dataset.targets = make_binary(
                self.train_dataset.data, self.train_dataset.targets, binary_labels
            )
            self.test_dataset.data, self.test_dataset.targets = make_binary(
                self.test_dataset.data, self.test_dataset.targets, binary_labels
            )
        if self.cfg.label_noise > 0:
            num_labels = 2 if self.cfg.binary_label else 10
            self.train_dataset.targets = add_label_noise(
                self.train_dataset.targets, self.cfg.label_noise, num_labels
            )
        self.x_train = torch.cat([x[0] for x in self.train_dataset])
        self.x_test = torch.cat([x[0] for x in self.test_dataset])
        self.y_train = torch.stack([x[1] for x in self.train_dataset])
        self.y_test = torch.stack([x[1] for x in self.test_dataset])
        self.x_train_id = torch.arange(self.x_train.shape[0])
        self.x_test_id = torch.arange(self.x_test.shape[0])
