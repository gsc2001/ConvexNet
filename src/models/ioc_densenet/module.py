from torch import nn, optim
import pytorch_lightning as pl
import numpy as np
import torchmetrics
import torch
from .model import IOCDenseNet


class IOCDensenetModule(pl.LightningModule):
    def __init__(self, growth_rate, block_config, num_init_features, lr, **kwargs):
        self.save_hyperparameters()
        super(IOCDensenetModule, self).__init__()

        self.model = IOCDenseNet(growth_rate, block_config, num_init_features, **kwargs)
        self.criterion = nn.CrossEntropyLoss()

        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4, momentum=0.9)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 10, 0.1)
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        self.train_acc(y_hat, y)
        loss = self.criterion(y_hat, y)

        self.log("train/acc", self.train_acc, on_epoch=True, prog_bar=True, logger=True)
        self.log("train/loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx, unused=0):
        for name, param in self.named_parameters():
            if ('weight' in name) and ('conv0' not in name):
                param_data = param.data.cpu().numpy()
                param_data[param_data < 0] = np.exp(
                    param_data[param_data < 0] - 5)
                param.data.copy_(torch.tensor(param_data))

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        self.valid_acc(y_hat, y)
        loss = self.criterion(y_hat, y)
        self.log("val/acc", self.valid_acc, on_epoch=True, prog_bar=True, logger=True)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss
