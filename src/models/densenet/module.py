from torch import nn, optim
import pytorch_lightning as pl
import torchmetrics
from .model import DenseNet


class DensenetModule(pl.LightningModule):
    def __init__(self, growth_rate, block_config, num_init_features, lr, epochs, **kwargs):
        self.save_hyperparameters()
        super(DensenetModule, self).__init__()

        self.model = DenseNet(growth_rate, block_config, num_init_features, **kwargs)
        self.criterion = nn.CrossEntropyLoss()

        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('Model')
        parser.add_argument('--lr', type=float, default=.1)
        return parent_parser

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4, momentum=0.9)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, self.hparams.epochs / 3, 0.1)
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        self.train_acc(y_hat, y)
        loss = self.criterion(y_hat, y)

        self.log("train/acc", self.train_acc, on_epoch=True, prog_bar=True, logger=True)
        self.log("train/loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        self.valid_acc(y_hat, y)
        loss = self.criterion(y_hat, y)
        self.log("val/acc", self.valid_acc, on_epoch=True, prog_bar=True, logger=True)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss
