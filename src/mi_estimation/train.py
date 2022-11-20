import argparse
import os

import wandb
from easydict import EasyDict
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from data import FashionMNISTDataModule, MNISTDataModule, SZTDataModule
from model import FCN


def parse_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    # Dataset
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["szt_g1", "szt_g2", "szt_var_u", "mnist", "fashionmnist"],
    )
    parser.add_argument("--label_noise", type=float, default=0.0)
    parser.add_argument("--binary_label", action="store_true")
    parser.add_argument("--image_size", type=int)
    # Model
    parser.add_argument("--layer_shapes", type=str, default="12x10x7x5x4x3x2")
    parser.add_argument(
        "--activation", type=str, default="tanh", choices=["relu", "tanh"]
    )
    # Training
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-1)
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        choices=["constant", "inverse_sqrt", "inverse_slow"],
    )
    parser.add_argument("--max_epochs", type=int, default=10000)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument(
        "--optimizer", type=str, default="adam", choices=["adam", "sgd"]
    )
    parser.add_argument("--num_workers", default="full")
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
    # Initialize model
    model = FCN(cfg)
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
    logger = WandbLogger(
        offline=not cfg.wandb,
        project=cfg.project_id,
        entity="ib-double-descent",
        config=cfg,
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
    )
    # Train and test
    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule)
    wandb.finish(quiet=True)


if __name__ == "__main__":
    main()
