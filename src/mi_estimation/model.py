import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning import LightningModule
from torchinfo import summary
from torchmetrics.functional import accuracy

from utils import calculate_layer_mi, grad_norm, weight_norm


class BaseModel(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

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
        self.log_dict(grad_norm(self), logger=True)

    def training_epoch_end(self, outputs):
        loss = torch.stack([i["loss"] for i in outputs]).double().mean()
        acc = torch.stack([i["train_acc"] for i in outputs]).double().mean()
        self.log_dict({"avg_train_acc": acc, "avg_train_loss": loss}, logger=True)
        self.log_dict(weight_norm(self), logger=True)

    def validation_step(self, batch, batch_idx):
        loss, acc = self.evaluate(batch, stage="val")
        return {"val_loss": loss, "val_acc": acc}

    def validation_epoch_end(self, outputs):
        loss = torch.stack([i["val_loss"] for i in outputs]).double().mean()
        acc = torch.stack([i["val_acc"] for i in outputs]).double().mean()
        self.log_dict({"avg_val_acc": acc, "avg_val_loss": loss}, logger=True)
        if self.config.log_mi:
            # estimate mutual information
            x_train = self.trainer.datamodule.x_train
            y_train = self.trainer.datamodule.y_train
            x_train_id = self.trainer.datamodule.x_train_id
            with torch.no_grad():
                _, Ts = self(x_train.to(self.device))
            for layer_idx, t in enumerate(Ts, 1):
                t = t.cpu()
                i_xt, i_yt = calculate_layer_mi(
                    t, self.config.num_bins, self.config.activation, x_train_id, y_train
                )
                self.log(f"l{layer_idx}_i_xt", i_xt, logger=True)
                self.log(f"l{layer_idx}_i_yt", i_yt, logger=True)

    def test_step(self, batch, batch_idx):
        loss, acc = self.evaluate(batch, "test")
        return {"test_loss": loss, "test_acc": acc}

    def test_epoch_end(self, outputs):
        acc = torch.stack([i["test_acc"] for i in outputs]).double().mean()
        loss = torch.stack([i["test_loss"] for i in outputs]).double().mean()
        self.log_dict({"avg_test_acc": acc, "avg_test_loss": loss}, logger=True)

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
            self._layers.append(layer)
            self.add_module(f"layer{i}", layer)
        # last layer
        self.fc = nn.Linear(layer_shapes[-2], layer_shapes[-1])
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
