import datetime
from math import sqrt

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from pytorch_lightning import LightningModule
from torchinfo import summary
from torchmetrics.functional import accuracy

from utils import (
    calculate_layer_mi,
    dict_average,
    grad_stats,
    log_now,
    plot_mi,
    weight_stats,
)


class BaseModel(LightningModule):
    """A lightning module that serves as a base class for all modules."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.train_acc_at_epoch = None
        self.grad_stats_per_step = []
        self.results = []

    def on_train_start(self):
        """Log model summary and hyperparameters."""
        input_size = self.trainer.datamodule.x_train.shape[1]
        model_summary = summary(
            self, input_size=(1, input_size), verbose=self.cfg.verbose
        )
        self.log_dict(
            {
                "total_params": float(model_summary.total_params),
                "trainable_params": float(model_summary.trainable_params),
            },
            logger=True,
        )

    def evaluate(self, batch, stage=None):
        """Evaluate the model on a batch. This is used for training, validation
        and testing."""
        x, y = batch
        output, _ = self(x)
        loss = F.cross_entropy(output, y)
        acc = accuracy(output.argmax(dim=1), y)
        self.log_dict({f"{stage}_loss": loss, f"{stage}_acc": acc}, logger=True)
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self.evaluate(batch, stage="train")
        return {"loss": loss, "train_acc": acc}

    def on_after_backward(self):
        """Log gradient stats."""
        if self.cfg.log_grad_stats:
            self.grad_stats_per_step.append(grad_stats(self))

    def training_epoch_end(self, outputs):
        loss = torch.stack([i["loss"] for i in outputs]).double().mean()
        acc = torch.stack([i["train_acc"] for i in outputs]).double().mean()
        self.log_dict({"avg_train_acc": acc, "avg_train_loss": loss}, logger=True)
        self.train_acc_at_epoch = acc

    def validation_step(self, batch, batch_idx):
        loss, acc = self.evaluate(batch, stage="val")
        return {"val_loss": loss, "val_acc": acc}

    def validation_epoch_end(self, outputs):
        loss = torch.stack([i["val_loss"] for i in outputs]).double().mean()
        acc = torch.stack([i["val_acc"] for i in outputs]).double().mean()
        self.log_dict({"avg_val_acc": acc, "avg_val_loss": loss}, logger=True)
        # Aggregate results
        should_log_now = log_now(self.current_epoch)
        # Log weight stats
        if should_log_now:
            epoch_results = {"epoch": self.current_epoch}
            if self.cfg.log_weight_stats:
                weight_stats_at_epoch = weight_stats(self)
                epoch_results.update(weight_stats_at_epoch)
            if self.cfg.log_grad_stats:
                grad_stats_at_epoch = dict_average(self.grad_stats_per_step)
                epoch_results.update(grad_stats_at_epoch)
                self.grad_stats_per_step = []
            # Estimate mutual information
            if self.cfg.log_mi:
                layer_mi_at_epoch = self.estimate_mi()
                epoch_results.update(layer_mi_at_epoch)
            epoch_results["train_acc"] = self.train_acc_at_epoch
            epoch_results["val_acc"] = acc
            self.results.append(epoch_results)

    def test_step(self, batch, batch_idx):
        loss, acc = self.evaluate(batch, "test")
        return {"test_loss": loss, "test_acc": acc}

    def test_epoch_end(self, outputs):
        acc = torch.stack([i["test_acc"] for i in outputs]).double().mean()
        loss = torch.stack([i["test_loss"] for i in outputs]).double().mean()
        self.log_dict({"avg_test_acc": acc, "avg_test_loss": loss}, logger=True)

    def on_train_end(self):
        if self.cfg.log_mi:
            # Save mutual information to csv and plot
            current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            title = (
                self.cfg.dataset
                + "_"
                + self.cfg.activation
                + "_"
                + self.cfg.layer_shapes
                + "_"
                + self.cfg.optimizer
            )
            filename = f"{current_time}_{title}"
            results = pd.DataFrame(self.results)
            results.to_csv(f"{filename}.csv", index=False)
            # Plot mutual information
            plot_mi(results, filename, self.cfg.layer_shapes.count("x"))
            # Save csv and plot to wandb
            wandb.save(f"{filename}.csv")
            wandb.save(f"{filename}.png")
            wandb.save(f"{filename}.pdf")
            wandb.log({"information_plane": wandb.Image(f"{filename}.png")})

    def configure_optimizers(self):
        if self.cfg.optimizer == "adam":
            optimizer = optim.Adam(self.parameters(), lr=self.cfg.lr)
        elif self.cfg.optimizer == "sgd":
            optimizer = optim.SGD(
                self.parameters(), lr=self.cfg.lr, momentum=self.cfg.momentum
            )
        return optimizer

    def estimate_mi(self):
        """Estimate mutual information."""
        x_train = self.trainer.datamodule.x_train
        y_train = self.trainer.datamodule.y_train
        x_train_id = self.trainer.datamodule.x_train_id
        x_test = self.trainer.datamodule.x_test
        y_test = self.trainer.datamodule.y_test
        x_test_id = self.trainer.datamodule.x_test_id
        with torch.no_grad():
            _, Ts_train = self(x_train.to(self.device))
            _, Ts_test = self(x_test.to(self.device))
        layer_mi_at_epoch = {}
        for idx, t in enumerate(Ts_train, 1):
            t = t.cpu()
            i_xt, i_ty = calculate_layer_mi(
                x_train_id, t, y_train, self.cfg.activation, self.cfg.num_bins
            )
            layer_mi_at_epoch[f"l{idx}_i_xt_tr"] = i_xt
            layer_mi_at_epoch[f"l{idx}_i_ty_tr"] = i_ty
        for idx, t in enumerate(Ts_test, 1):
            t = t.cpu()
            i_xt, i_ty = calculate_layer_mi(
                x_test_id, t, y_test, self.cfg.activation, self.cfg.num_bins
            )
            layer_mi_at_epoch[f"l{idx}_i_xt_te"] = i_xt
            layer_mi_at_epoch[f"l{idx}_i_ty_te"] = i_ty
        return layer_mi_at_epoch


class FCN(BaseModel):
    """Fully connected neural network."""

    def __init__(self, cfg):
        super().__init__(cfg)
        layer_shapes = [int(x) for x in cfg.layer_shapes.split("x")]
        self._layers = []
        # Input layer and hidden layers
        for i in range(1, len(layer_shapes) - 1):
            layer = nn.Linear(layer_shapes[i - 1], layer_shapes[i])
            nn.init.trunc_normal_(layer.weight, mean=0, std=sqrt(1 / layer_shapes[i]))
            nn.init.zeros_(layer.bias)
            self._layers.append(layer)
            self.add_module(f"layer{i}", layer)
        # Last layer
        self.fc = nn.Linear(layer_shapes[-2], layer_shapes[-1])
        nn.init.trunc_normal_(self.fc.weight, mean=0, std=sqrt(1 / layer_shapes[-1]))
        nn.init.zeros_(self.fc.bias)
        # Choose activation function
        if cfg.activation == "relu":
            self.activation = torch.relu
        elif cfg.activation == "tanh":
            self.activation = torch.tanh

    def forward(self, x):
        if x.dim() > 2:
            batch_size = x.shape[0]
            x = x.view(batch_size, -1)
        Ts = []  # intermediate outputs for all layers
        for layer in self._layers:
            x = self.activation(layer(x))
            Ts.append(x.clone().detach())
        x = self.fc(x)
        Ts.append(self.activation(x).clone().detach())
        return x, Ts
