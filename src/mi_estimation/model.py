import datetime
from math import sqrt

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from pytorch_lightning import LightningModule
from torchinfo import summary
from torchmetrics.functional import accuracy

from utils import calculate_layer_mi, grad_norm, log_now, weight_norm


class BaseModel(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer_i_xt = []
        self.layer_i_yt = []

    def on_train_start(self):
        input_size = self.trainer.datamodule.x_train.shape[1]
        model_summary = summary(
            self, input_size=(1, input_size), verbose=self.config.verbose
        )
        self.log_dict(
            {
                "total_params": float(model_summary.total_params),
                "trainable_params": float(model_summary.trainable_params),
            },
            logger=True,
        )

    def evaluate(self, batch, stage=None):
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
        if self.config.log_grad_norm:
            self.log_dict(grad_norm(self), logger=True)

    def training_epoch_end(self, outputs):
        loss = torch.stack([i["loss"] for i in outputs]).double().mean()
        acc = torch.stack([i["train_acc"] for i in outputs]).double().mean()
        self.log_dict({"avg_train_acc": acc, "avg_train_loss": loss}, logger=True)
        if self.config.log_weight_norm:
            self.log_dict(weight_norm(self), logger=True)
        if self.config.log_mi and log_now(self.current_epoch):
            # estimate mutual information
            x_train = self.trainer.datamodule.x_train
            y_train = self.trainer.datamodule.y_train
            x_train_id = self.trainer.datamodule.x_train_id
            with torch.no_grad():
                _, Ts = self(x_train.to(self.device))
            layer_i_xt_at_epoch, layer_i_yt_at_epoch = {}, {}
            layer_i_xt_at_epoch["epoch"] = self.current_epoch
            layer_i_yt_at_epoch["epoch"] = self.current_epoch
            for layer_idx, t in enumerate(Ts, 1):
                t = t.cpu()
                i_xt, i_yt = calculate_layer_mi(
                    t, self.config.num_bins, self.config.activation, x_train_id, y_train
                )
                layer_i_xt_at_epoch[f"l{layer_idx}_i_xt"] = i_xt
                layer_i_yt_at_epoch[f"l{layer_idx}_i_yt"] = i_yt
            self.layer_i_xt.append(layer_i_xt_at_epoch)
            self.layer_i_yt.append(layer_i_yt_at_epoch)

    def validation_step(self, batch, batch_idx):
        loss, acc = self.evaluate(batch, stage="val")
        return {"val_loss": loss, "val_acc": acc}

    def validation_epoch_end(self, outputs):
        loss = torch.stack([i["val_loss"] for i in outputs]).double().mean()
        acc = torch.stack([i["val_acc"] for i in outputs]).double().mean()
        self.log_dict({"avg_val_acc": acc, "avg_val_loss": loss}, logger=True)

    def test_step(self, batch, batch_idx):
        loss, acc = self.evaluate(batch, "test")
        return {"test_loss": loss, "test_acc": acc}

    def test_epoch_end(self, outputs):
        acc = torch.stack([i["test_acc"] for i in outputs]).double().mean()
        loss = torch.stack([i["test_loss"] for i in outputs]).double().mean()
        self.log_dict({"avg_test_acc": acc, "avg_test_loss": loss}, logger=True)

    def on_train_end(self):
        if self.config.log_mi:
            # save mutual information to csv and plot
            current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            df_i_xt = pd.DataFrame(self.layer_i_xt)
            df_i_xt.to_csv(f"i_xt_{current_time}.csv", index=False)
            df_i_yt = pd.DataFrame(self.layer_i_yt)
            df_i_yt.to_csv(f"i_yt_{current_time}.csv", index=False)
            wandb.log({"i_xt": wandb.Table(dataframe=df_i_xt)})
            wandb.log({"i_yt": wandb.Table(dataframe=df_i_yt)})
            # plot information plane
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.set_xlabel(r"$I(X; T)$")
            ax.set_ylabel(r"$I(Y; T)$")
            num_layers = self.config.layer_shapes.count("x")
            for i in range(num_layers):
                a = ax.scatter(
                    df_i_xt[f"l{i+1}_i_xt"],
                    df_i_yt[f"l{i+1}_i_yt"],
                    s=50,
                    c=df_i_xt["epoch"],
                    cmap="viridis",
                    # norm=colors.LogNorm(vmin=1, vmax=df_i_xt["epoch"].max()),
                )
            fig.colorbar(a, label="Epoch")
            fig.savefig(f"information_plane_{current_time}.pdf", bbox_inches="tight")
            fig.savefig(
                f"information_plane_{current_time}.png", bbox_inches="tight", dpi=600
            )
            wandb.log(
                {
                    "information_plane": wandb.Image(
                        f"information_plane_{current_time}.png"
                    )
                }
            )

    def configure_optimizers(self):
        if self.config.optimizer == "adam":
            optimizer = optim.Adam(self.parameters(), lr=self.config.lr)
        elif self.config.optimizer == "sgd":
            optimizer = optim.SGD(
                self.parameters(), lr=self.config.lr, momentum=self.config.momentum
            )
        return optimizer


class FCN(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        layer_shapes = [int(x) for x in config.layer_shapes.split("x")]
        self._layers = []
        for i in range(1, len(layer_shapes) - 1):
            layer = nn.Linear(layer_shapes[i - 1], layer_shapes[i])
            nn.init.trunc_normal_(layer.weight, mean=0, std=sqrt(1 / layer_shapes[i]))
            nn.init.zeros_(layer.bias)
            self._layers.append(layer)
            self.add_module(f"layer{i}", layer)
        # last layer
        self.fc = nn.Linear(layer_shapes[-2], layer_shapes[-1])
        nn.init.trunc_normal_(self.fc.weight, mean=0, std=sqrt(1 / layer_shapes[-1]))
        nn.init.zeros_(self.fc.bias)
        # choose activation function
        if config.activation == "relu":
            self.activation = torch.relu
        elif config.activation == "tanh":
            self.activation = torch.tanh
        elif config.activation == "sigmoid":
            self.activation = torch.sigmoid

    def forward(self, x):
        Ts = []  # intermediate outputs
        for layer in self._layers:
            x = self.activation(layer(x))
            Ts.append(x.clone().detach())
        x = self.fc(x)
        Ts.append(x.clone().detach())
        return x, Ts
