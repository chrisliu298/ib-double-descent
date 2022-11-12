import torch
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning import LightningModule
from torchinfo import summary
from torchmetrics.functional import accuracy

from utils import grad_norm, lr_schedules, weight_norm


class BaseModel(LightningModule):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.config = config

    def forward(self, x):
        return self.model(x)

    def on_train_start(self):
        # log model summary
        wh = self.trainer.datamodule.wh
        ch = self.trainer.datamodule.ch
        if self.config.model == "cnn":
            assert int(self.config.layer_config.split("x")[0]) == ch
        elif self.config.model == "fcn":
            assert int(self.config.layer_config.split("x")[0]) == wh * wh * ch
        model_summary = summary(
            self, input_size=(1, ch, wh, wh), verbose=self.config.verbose
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
        y_hat = self(x)
        if self.config.loss == "ce":
            loss = F.cross_entropy(y_hat, y)
        elif self.config.loss == "mse":
            loss = F.mse_loss(y_hat, y)
            y = y.argmax(dim=1)
        acc = accuracy(y_hat.argmax(dim=1), y)
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

    def test_step(self, batch, batch_idx):
        loss, acc = self.evaluate(batch, "test")
        return {"test_loss": loss, "test_acc": acc}

    def test_epoch_end(self, outputs):
        acc = torch.stack([i["test_acc"] for i in outputs]).double().mean()
        loss = torch.stack([i["test_loss"] for i in outputs]).double().mean()
        self.log_dict({"avg_test_acc": acc, "avg_test_loss": loss}, logger=True)

    def configure_optimizers(self):
        if self.config.optimizer == "adam":
            optimizer = optim.AdamW(
                self.parameters(), lr=self.config.lr, weight_decay=self.config.wd
            )
        elif self.config.optimizer == "sgd":
            optimizer = optim.SGD(
                self.parameters(),
                lr=self.config.lr,
                weight_decay=self.config.wd,
                momentum=self.config.momentum,
            )
        if self.config.lr_sch is not None:
            sch = optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lr_schedules[self.config.lr_sch]
            )
            interval = "step" if self.config.lr_sch == "inverse" else "epoch"
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": sch,
                    "interval": interval,
                    "frequency": 1,
                },
            }
        return optimizer
