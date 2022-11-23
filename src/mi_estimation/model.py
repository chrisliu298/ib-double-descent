from math import sqrt

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from pytorch_lightning import LightningModule
from torchmetrics.functional import accuracy

from utils import (
    calculate_layer_mi,
    dict_average,
    grad_stats,
    log_now,
    lr_schedule,
    plot_mi,
    weight_stats,
)


class BaseModel(LightningModule):
    """A lightning module that serves as a base class for all modules."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.grad_stats_per_step = []
        self.epoch_results = {}
        self.results = []
        self.should_log_now = False

    def on_train_start(self):
        self.trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        self.total_params = sum(p.numel() for p in self.parameters())
        datamodule = self.trainer.datamodule
        self.log("total_params", float(self.total_params), logger=True)
        self.log("trainable_params", float(self.trainable_params), logger=True)
        self.log("train_size", float(len(datamodule.train_dataset)), logger=True)
        self.log("test_size", float(len(datamodule.test_dataset)), logger=True)

    def evaluate(self, batch, stage=None):
        """Evaluate the model on a batch. This is used for training, validation
        and testing."""
        x, y = batch
        output, _ = self(x)
        if self.cfg.loss == "ce":
            loss = F.cross_entropy(output, y)
        elif self.cfg.loss == "mse":
            loss = F.mse_loss(output, y)
            y = y.argmax(dim=1)
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
        # Aggregate results
        should_log_now = log_now(self.current_epoch + 1)
        should_log_now = should_log_now or self.current_epoch == self.cfg.max_epochs - 1
        self.should_log_now = should_log_now
        # Log weight stats
        if self.should_log_now:
            self.epoch_results = {"epoch": self.current_epoch}
            if self.cfg.log_weight_stats:
                weight_stats_at_epoch = weight_stats(self)
                self.epoch_results.update(weight_stats_at_epoch)
            if self.cfg.log_grad_stats:
                grad_stats_at_epoch = dict_average(self.grad_stats_per_step)
                self.epoch_results.update(grad_stats_at_epoch)
                self.grad_stats_per_step = []
            # Estimate mutual information
            if self.cfg.log_mi:
                layer_mi_at_epoch = self.estimate_mi()
                self.epoch_results.update(layer_mi_at_epoch)
            self.epoch_results["total_params"] = (
                self.trainable_params
                if self.trainable_params != self.total_params
                else self.total_params
            )
            self.epoch_results["train_acc"] = acc.item()
            self.epoch_results["train_loss"] = loss.item()

    def validation_step(self, batch, batch_idx):
        loss, acc = self.evaluate(batch, stage="val")
        return {"val_loss": loss, "val_acc": acc}

    def validation_epoch_end(self, outputs):
        loss = torch.stack([i["val_loss"] for i in outputs]).double().mean()
        acc = torch.stack([i["val_acc"] for i in outputs]).double().mean()
        self.log_dict({"avg_val_acc": acc, "avg_val_loss": loss}, logger=True)
        if self.should_log_now:
            self.epoch_results["test_acc"] = acc.item()
            self.epoch_results["test_loss"] = loss.item()
            self.results.append(self.epoch_results)

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
            title = (
                self.cfg.dataset
                + "_"
                + self.cfg.arch
                + "_"
                + self.cfg.layer_dims
                + "_"
                + self.cfg.activation
                + "_"
                + self.cfg.optimizer
                + "_"
                + self.cfg.loss
                + "_"
                + str(self.total_params)
            )
            results = pd.DataFrame(self.results)
            results.to_csv(f"{title}.csv", index=False)
            # Plot mutual information
            plot_mi(results, title, self.cfg.layer_dims.count("x"))
            # Save csv and plot to wandb
            wandb.save(f"{title}.csv")
            wandb.save(f"{title}.png")
            wandb.save(f"{title}.pdf")
            wandb.log({"information_plane": wandb.Image(f"{title}.png")})

    def configure_optimizers(self):
        if self.cfg.optimizer == "adam":
            optimizer = optim.Adam(self.parameters(), lr=self.cfg.lr)
        elif self.cfg.optimizer == "sgd":
            optimizer = optim.SGD(
                self.parameters(), lr=self.cfg.lr, momentum=self.cfg.momentum
            )
        lr_scheduler = optim.lr_scheduler.LambdaLR(
            optimizer, lr_schedule(self.cfg.lr_scheduler)
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "epoch",
                "frequency": self.cfg.lr_scheduler_freq,
            },
        }

    def estimate_mi(self):
        """Estimate mutual information."""
        x_train = self.trainer.datamodule.x_train
        y_train = self.trainer.datamodule.y_train
        x_train_id = self.trainer.datamodule.x_train_id
        # x_test = self.trainer.datamodule.x_test
        # y_test = self.trainer.datamodule.y_test
        # x_test_id = self.trainer.datamodule.x_test_id
        x_train = x_train.unsqueeze(1)
        # x_test = x_test.unsqueeze(1)
        if y_train.dim() == 2:
            y_train = y_train.argmax(dim=1)
        # if y_test.dim() == 2:
        #     y_test = y_test.argmax(dim=1)
        with torch.no_grad():
            _, Ts_train = self(x_train.to(self.device))
            # _, Ts_test = self(x_test.to(self.device))
        layer_mi_at_epoch = {}
        for idx, t in enumerate(Ts_train, 1):
            t = t.cpu()
            i_xt, i_ty = calculate_layer_mi(
                x_train_id, t, y_train, self.cfg.activation, self.cfg.num_bins
            )
            layer_mi_at_epoch[f"l{idx}_i_xt_tr"] = i_xt
            layer_mi_at_epoch[f"l{idx}_i_ty_tr"] = i_ty
        # for idx, t in enumerate(Ts_test, 1):
        #     t = t.cpu()
        #     i_xt, i_ty = calculate_layer_mi(
        #         x_test_id, t, y_test, self.cfg.activation, self.cfg.num_bins
        #     )
        #     layer_mi_at_epoch[f"l{idx}_i_xt_te"] = i_xt
        #     layer_mi_at_epoch[f"l{idx}_i_ty_te"] = i_ty
        return layer_mi_at_epoch


class RFN(BaseModel):
    """A random feature network."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        assert (
            self.cfg.layer_dims.count("x") == 2
        ), f"RFN must have 2 hidden layers, got {self.cfg.layer_dims.count('x')}"
        layer_dims = [int(x) for x in cfg.layer_dims.split("x")]
        in_features, hidden_features, out_features = layer_dims
        # Define layers
        self.layer0 = nn.Linear(in_features, hidden_features, bias=False)
        # nn.init.normal_(self.layer0.weight, mean=0, std=sqrt(1 / hidden_features))
        # nn.init.zeros_(self.layer0.bias)
        self.layer1 = nn.Linear(hidden_features, out_features, bias=False)
        # nn.init.normal_(self.layer1.weight, mean=0, std=sqrt(1 / out_features))
        # nn.init.zeros_(self.layer1.bias)
        # Freeze first layer
        self.layer0.weight.requires_grad = False
        self.layer0.bias.requires_grad = False
        # Choose activation function
        if cfg.activation == "relu":
            self.activation = torch.relu
        elif cfg.activation == "tanh":
            self.activation = torch.tanh

    def forward(self, x):
        x = x.flatten(1)
        Ts = []  # intermediate outputs for all layers
        x = self.layer0(x)
        x = self.activation(x)
        Ts.append(x.clone().detach())
        x = self.layer1(x)
        Ts.append(self.activation(x).clone().detach())
        return x, Ts


class FCN(BaseModel):
    """A fully connected neural network."""

    def __init__(self, cfg):
        super().__init__(cfg)
        layer_dims = [int(x) for x in cfg.layer_dims.split("x")]
        self._layers = []
        # Input layer and hidden layers
        for i in range(1, len(layer_dims) - 1):
            layer = nn.Linear(layer_dims[i - 1], layer_dims[i])
            # nn.init.trunc_normal_(layer.weight, mean=0, std=sqrt(1 / layer_dims[i]))
            # nn.init.zeros_(layer.bias)
            self._layers.append(layer)
            self.add_module(f"layer{i}", layer)
        # Last layer
        layer = nn.Linear(layer_dims[-2], layer_dims[-1])
        # nn.init.trunc_normal_(layer.weight, mean=0, std=sqrt(1 / layer_dims[-1]))
        # nn.init.zeros_(layer.bias)
        self._layers.append(layer)
        self.add_module(f"layer{len(layer_dims) - 1}", layer)
        # Choose activation function
        if cfg.activation == "relu":
            self.activation = torch.relu
        elif cfg.activation == "tanh":
            self.activation = torch.tanh

    def forward(self, x):
        x = x.flatten(1)
        Ts = []  # intermediate outputs for all layers
        for layer in self._layers[:-1]:
            x = self.activation(layer(x))
            Ts.append(x.clone().detach())
        x = self._layers[-1](x)
        Ts.append(self.activation(x).clone().detach())
        return x, Ts


class CNN(BaseModel):
    """A convolutional neural network."""

    def __init__(self, cfg):
        super().__init__(cfg)
        layer_dims = [int(x) for x in cfg.layer_dims.split("x")]
        self._layers = []
        # Input layer and hidden layers
        for i in range(1, len(layer_dims) - 1):
            layer = nn.Conv2d(
                layer_dims[i - 1], layer_dims[i], kernel_size=3, padding=1
            )
            # nn.init.trunc_normal_(
            #     layer.weight, mean=0, std=sqrt(1 / (layer_dims[i] * 3 * 3))
            # )
            # nn.init.zeros_(layer.bias)
            self._layers.append(layer)
            self.add_module(f"conv{i}", layer)
        # Last layer
        layer = nn.Conv2d(layer_dims[-2], layer_dims[-1], kernel_size=3, padding=1)
        # nn.init.trunc_normal_(
        #     layer.weight, mean=0, std=sqrt(1 / (layer_dims[-1] * 3 * 3))
        # )
        # nn.init.zeros_(layer.bias)
        self._layers.append(layer)
        self.add_module(f"conv{len(layer_dims) - 1}", layer)
        # Choose activation function
        if cfg.activation == "relu":
            self.activation = torch.relu
        elif cfg.activation == "tanh":
            self.activation = torch.tanh

    def forward(self, x):
        Ts = []  # intermediate outputs for all layers
        for i, layer in enumerate(self._layers[:-1]):
            x = self.activation(layer(x))
            Ts.append(x.clone().detach().flatten(1))
            if i > 0:
                x = F.max_pool2d(x, 2)
        x = F.max_pool2d(x, 4)
        x = x.flatten(1)
        x = self._layers[-1](x)
        Ts.append(self.activation(x).clone().detach().flatten(1))
        return x, Ts
