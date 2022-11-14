import argparse
import os

import wandb
from easydict import EasyDict
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from data import SZTDataModule
from model import FCN


def parse_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    # Dataset
    parser.add_argument("--dataset_path", type=str, required=True)
    # Model
    parser.add_argument("--layer_shapes", type=str, default="12x10x7x5x4x3x2")
    parser.add_argument(
        "--activation", type=str, default="tanh", choices=["relu", "tanh", "sigmoid"]
    )
    # Training
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-1)
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
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--log_mi", action="store_true")
    parser.add_argument("--log_grad_stats", action="store_true")
    parser.add_argument("--log_weight_stats", action="store_true")
    # Convert to an easydict object
    config = EasyDict(vars(parser.parse_args()))
    return config


def main():
    config = parse_args()
    # Set num workers
    if config.num_workers == "full":
        config.num_workers = os.cpu_count()
    # Set seed
    if config.seed is not None:
        seed_everything(config.seed)
    # Initialize data module
    datamodule = SZTDataModule(config)
    # Initialize model
    model = FCN(config)
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
        offline=not config.wandb,
        project=config.project_id,
        entity="ib-double-descent",
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
    # Train and test
    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule, verbose=config.verbose > 0)
    wandb.finish(quiet=True)


if __name__ == "__main__":
    main()
