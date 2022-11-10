import argparse

import torch
import wandb
from easydict import EasyDict
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from data import (
    CIFAR10DataModule,
    CIFAR100DataModule,
    FashionMNISTDataModule,
    MNISTDataModule,
)
from model import BaseModel
from models import *

DATASETS = {
    "mnist": MNISTDataModule,
    "fashionmnist": FashionMNISTDataModule,
    "cifar10": CIFAR10DataModule,
    "cifar100": CIFAR100DataModule,
}
MODELS = {"fcn": FCN, "cnn": CNN}
ACTVATIONS = {"relu": torch.relu, "tanh": torch.tanh, "sigmoid": torch.sigmoid}


def parse_args():
    # parse command line arguments
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument("--dataset", type=str, required=True, choices=DATASETS.keys())
    parser.add_argument("--project_id", type=str, default="dnn-mi")
    # model
    parser.add_argument("--model", type=str, required=True, choices=MODELS.keys())
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--input_size", type=int, default=784)
    parser.add_argument("--output_size", type=int, default=10)
    # parser.add_argument("--layer_config", type=str, default="768x512x256x128x10")
    parser.add_argument(
        "--activation", type=str, default="tanh", choices=["relu", "tanh", "sigmoid"]
    )
    # training
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--lr_sch", type=str, choices=["inverse", "inverse_slow", "inverse_sqrt"]
    )
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument(
        "--optimizer", type=str, default="adam", choices=["adam", "sgd"]
    )
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=0)
    # experiment
    parser.add_argument("--seed", type=int)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--wandb", action="store_true")
    # convert to an easydict object
    config = EasyDict(vars(parser.parse_args()))
    # assign layer config
    config.layer_config = "x".join(
        [str(config.input_size)]
        + [str(config.width)] * (config.depth - 1)
        + [str(config.output_size)]
    )
    return config


def main():
    config = parse_args()
    # set seed
    if config.seed is not None:
        seed_everything(config.seed)
    # initialize data module
    datamodule = DATASETS[config.dataset](config)
    # initialize model
    activation_fn = ACTVATIONS[config.activation]
    net = MODELS[config.model](config.layer_config, activation_fn)
    model = BaseModel(net, config)
    # setup trainer
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            filename="{epoch}-{avg_train_acc:.4f}-{avg_val_acc:.4f}",
            monitor="epoch",
            save_top_k=5,
            mode="max",
        ),
    ]
    logger = WandbLogger(
        offline=not config.wandb,
        project=config.project_id,
        entity="chrisliu298",
        config=config,
    )
    trainer = Trainer(
        accelerator="gpu",
        devices=-1,
        callbacks=callbacks,
        max_epochs=config.max_epochs,
        check_val_every_n_epoch=1,
        benchmark=True,
        logger=logger,
        enable_progress_bar=config.verbose > 0,
    )
    # train and test
    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule, verbose=config.verbose > 0)
    wandb.finish(quiet=True)


if __name__ == "__main__":
    main()
