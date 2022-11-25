import argparse
import os
import sys
import json
from typing import Optional

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
    SZTDataModule,
)
from model import CNN, FCN, RFN


def parse_args():
    # An alternative way to define and use arguments: define all as JSON and just load
    # from file. Only used if first CLI argument is an existing file (presumed JSON format)
    # `python train.py folder/my_run_spec.json`
    #
    # (e.g. if running a bunch of variants from a folder)
    if os.path.exists(sys.argv[1]):
        run_file: str = sys.argv[1]
        with open(run_file, 'r') as f:
            cfg = EasyDict(json.load(f))
            return cfg

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    # Dataset
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=[
            "szt_g1",
            "szt_g2",
            "szt_var_u",
            "mnist",
            "fashionmnist",
            "cifar10",
            "cifar100",
        ],
    )
    parser.add_argument("--train_size", type=int, default=-1)
    parser.add_argument("--label_noise", type=float, default=0.0)
    parser.add_argument("--binary_label", action="store_true")
    parser.add_argument("--image_size", type=int)
    # Model
    parser.add_argument(
        "--arch", type=str, required=True, choices=["rfn", "fcn", "cnn"]
    )
    parser.add_argument("--layer_dims", type=str, default="12x10x7x5x4x3x2")
    parser.add_argument(
        "--activation", type=str, default="tanh", choices=["relu", "tanh"]
    )
    # Training
    parser.add_argument("--loss", type=str, default="ce", choices=["ce", "mse"])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-1)
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        choices=["constant", "inverse_sqrt", "inverse"],
    )
    parser.add_argument("--lr_scheduler_freq", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=10000)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument(
        "--optimizer", type=str, default="adam", choices=["adam", "sgd"]
    )
    parser.add_argument("--num_workers", default="full")
    parser.add_argument("--work_dir", default=None)
    # MI estimation
    parser.add_argument("--num_bins", type=int, default=30)
    # Experiment
    parser.add_argument("--project_id", type=str, default="dnn-mi")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--log_mi", action="store_true")
    parser.add_argument("--log_grad_stats", action="store_true")
    parser.add_argument("--log_weight_stats", action="store_true")
    # Convert to an easydict object
    cfg = EasyDict(vars(parser.parse_args()))
    return cfg


def main():
    cfg = parse_args()
    # Set num workers
    if cfg.num_workers == "full":
        cfg.num_workers = os.cpu_count()
    # Set seed
    if cfg.seed is not None:
        seed_everything(cfg.seed)
    # Initialize data module
    if "szt_" in cfg.dataset:
        datamodule = SZTDataModule(cfg)
    elif cfg.dataset == "mnist":
        datamodule = MNISTDataModule(cfg)
    elif cfg.dataset == "fashionmnist":
        datamodule = FashionMNISTDataModule(cfg)
    elif cfg.dataset == "cifar10":
        datamodule = CIFAR10DataModule(cfg)
    elif cfg.dataset == "cifar100":
        datamodule = CIFAR100DataModule(cfg)
    # Initialize model
    if cfg.arch == "fcn":
        model = FCN(cfg)
    elif cfg.arch == "cnn":
        model = CNN(cfg)
    elif cfg.arch == "rfn":
        model = RFN(cfg)
    # Setup trainer
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            filename="{epoch}-{avg_train_acc:.4f}-{avg_val_acc:.4f}",
            monitor="epoch",
            save_top_k=5,
            mode="max",
        ),
    ]
    # default is None, uses current working directory if set to None on logger
    work_dir: Optional[str] = cfg.get('work_dir', None)
    if work_dir and not os.path.exists(work_dir):
        os.makedirs(work_dir, exist_ok=False)
    logger = WandbLogger(
        offline=not cfg.wandb,
        project=cfg.project_id,
        entity="ib-double-descent",
        config=cfg,
        dir=work_dir
    )
    trainer = Trainer(
        accelerator="gpu",
        devices=-1,
        callbacks=callbacks,
        max_epochs=cfg.max_epochs,
        check_val_every_n_epoch=1,
        benchmark=True,
        logger=logger,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
    )
    # Train and test
    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule, verbose=False)
    wandb.finish(quiet=True)


if __name__ == "__main__":
    main()
