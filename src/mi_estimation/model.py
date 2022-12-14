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
    activation_fn,
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
        self.history = {}

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
        should_log_now = log_now(self.current_epoch) or (
            self.current_epoch + 1 == self.cfg.max_epochs
        )
        if should_log_now:
            self.history[self.current_epoch]["train_loss"] = loss.item()
            self.history[self.current_epoch]["train_acc"] = acc.item()

    def validation_step(self, batch, batch_idx):
        loss, acc = self.evaluate(batch, stage="val")
        return {"val_loss": loss, "val_acc": acc}

    def validation_epoch_end(self, outputs):
        loss = torch.stack([i["val_loss"] for i in outputs]).double().mean()
        acc = torch.stack([i["val_acc"] for i in outputs]).double().mean()
        self.log_dict({"avg_val_acc": acc, "avg_val_loss": loss}, logger=True)
        should_log_now = log_now(self.current_epoch) or (
            self.current_epoch + 1 == self.cfg.max_epochs
        )
        if should_log_now:
            self.history[self.current_epoch] = {}
            self.history[self.current_epoch]["epoch"] = self.current_epoch + 1
            if self.cfg.log_weight_stats:
                weight_stats_at_epoch = weight_stats(self)
                self.history[self.current_epoch].update(weight_stats_at_epoch)
            if self.cfg.log_grad_stats:
                grad_stats_at_epoch = dict_average(self.grad_stats_per_step)
                self.history[self.current_epoch].update(grad_stats_at_epoch)
                self.grad_stats_per_step = []
            # Estimate mutual information
            if self.cfg.log_mi:
                layer_mi_at_epoch = self.estimate_mi()
                self.history[self.current_epoch].update(layer_mi_at_epoch)
            self.history[self.current_epoch]["total_params"] = self.total_params
            self.history[self.current_epoch]["trainable_params"] = self.trainable_params
            self.history[self.current_epoch]["val_loss"] = loss.item()
            self.history[self.current_epoch]["val_acc"] = acc.item()

    def test_step(self, batch, batch_idx):
        loss, acc = self.evaluate(batch, "test")
        return {"test_loss": loss, "test_acc": acc}

    def test_epoch_end(self, outputs):
        acc = torch.stack([i["test_acc"] for i in outputs]).double().mean()
        loss = torch.stack([i["test_loss"] for i in outputs]).double().mean()
        self.log_dict({"avg_test_acc": acc, "avg_test_loss": loss}, logger=True)

    def on_train_end(self):
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
            + str(self.trainable_params)
        )
        results = [v for k, v in self.history.items()]
        results = sorted(results, key=lambda i: i["epoch"])
        results = pd.DataFrame(results)
        results = results.reindex(sorted(results.columns), axis=1)
        results.to_csv(f"{title}.csv", index=False)
        wandb.save(f"{title}.csv")
        # Plot mutual information
        if self.cfg.log_mi:
            plot_mi(results, title, self.cfg.layer_dims.count("x"))
            wandb.log({"information_plane": wandb.Image(f"{title}.png")})
            wandb.save(f"{title}.png")
            wandb.save(f"{title}.pdf")

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
        nn.init.normal_(self.layer0.weight, mean=0, std=sqrt(1 / in_features))
        self.layer1 = nn.Linear(hidden_features, out_features, bias=False)
        nn.init.normal_(self.layer1.weight, mean=0, std=sqrt(1 / hidden_features))
        # Freeze first layer
        self.layer0.weight.requires_grad = False
        # Choose activation function
        self.activation = activation_fn(cfg.activation)

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
        self.activation = activation_fn(cfg.activation)

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
        self.activation = activation_fn(cfg.activation)

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
