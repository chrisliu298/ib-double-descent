import os

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, FashionMNIST


class BaseDataModule(LightningDataModule):
    def __init__(self):
        super().__init__()
        self.num_workers = os.cpu_count()

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


class MNISTDataModule(BaseDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # define x and y transforms
        self.x_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        self.y_transform = None
        self.wh = 28
        self.ch = 1

    def prepare_data(self):
        # download data
        MNIST("/tmp/data", train=True, download=True)
        MNIST("/tmp/data", train=False, download=True)

    def setup(self, stage=None):
        # load (downloaded) data
        self.train_dataset = MNIST(
            "/tmp/data",
            train=True,
            transform=self.x_transform,
            target_transform=self.y_transform,
        )
        self.test_dataset = MNIST(
            "/tmp/data",
            train=False,
            transform=self.x_transform,
            target_transform=self.y_transform,
        )


class FashionMNISTDataModule(BaseDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # define x and y transforms
        self.x_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.2860,), (0.3530,)),
            ]
        )
        self.y_transform = None
        self.wh = 28
        self.ch = 1

    def prepare_data(self):
        # download data
        FashionMNIST("/tmp/data", train=True, download=True)
        FashionMNIST("/tmp/data", train=False, download=True)

    def setup(self, stage=None):
        # load (downloaded) data
        self.train_dataset = FashionMNIST(
            "/tmp/data",
            train=True,
            transform=self.x_transform,
            target_transform=self.y_transform,
        )
        self.test_dataset = FashionMNIST(
            "/tmp/data",
            train=False,
            transform=self.x_transform,
            target_transform=self.y_transform,
        )


class CIFAR10DataModule(BaseDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # define x and y transforms
        self.x_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
                ),
            ]
        )
        self.y_transform = None
        self.wh = 32
        self.ch = 3

    def prepare_data(self):
        # download data
        CIFAR10("/tmp/data", train=True, download=True)
        CIFAR10("/tmp/data", train=False, download=True)

    def setup(self, stage=None):
        # load (downloaded) data
        self.train_dataset = CIFAR10(
            "/tmp/data",
            train=True,
            transform=self.x_transform,
            target_transform=self.y_transform,
        )
        self.test_dataset = CIFAR10(
            "/tmp/data",
            train=False,
            transform=self.x_transform,
            target_transform=self.y_transform,
        )


class CIFAR100DataModule(BaseDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # define x and y transforms
        self.x_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5071, 0.4867, 0.4408), (0.2673, 0.2564, 0.2762)
                ),
            ]
        )
        self.y_transform = None
        self.wh = 32
        self.ch = 3

    def prepare_data(self):
        # download data
        CIFAR100("/tmp/data", train=True, download=True)
        CIFAR100("/tmp/data", train=False, download=True)

    def setup(self, stage=None):
        # load (downloaded) data
        self.train_dataset = CIFAR100(
            "/tmp/data",
            train=True,
            transform=self.x_transform,
            target_transform=self.y_transform,
        )
        self.test_dataset = CIFAR100(
            "/tmp/data",
            train=False,
            transform=self.x_transform,
            target_transform=self.y_transform,
        )
