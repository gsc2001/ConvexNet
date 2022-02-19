import pytorch_lightning as pl
import torchmetrics as metrics
from torch import optim
import torch.nn.functional as F
from timm.data import Mixup
from .model import MlpMixer
from ignite.handlers import create_lr_scheduler_with_warmup
from timm.scheduler import PolyLRScheduler
from utils import mixup_data, mixup_criterion


class MixerModule(pl.LightningModule):
    def __init__(self, model: MlpMixer, lr=1e-3, wd=.1, lr_warmup_epochs=35, n_epochs=300,
                 mixup_p=.5):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.mixup_fn = Mixup(mixup_alpha=mixup_p)
        self.train_accuracy = metrics.Accuracy()
        self.valid_accuracy = metrics.Accuracy()

    def forward(self, x):
        y = self.model(x)
        return y

    def training_step(self, batch, batch_idx):
        x, y = batch
        mixed_x, mixed_y = self.mixup_fn(x, y)
        y_hat = self(mixed_x)
        loss = F.cross_entropy(y_hat, mixed_y)
        self.train_accuracy(y_hat.argmax(1).int(), mixed_y.argmax(1).int())
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,sync_dist=True)
        self.log("train_acc", self.train_accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True,sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.valid_accuracy(y_hat, y)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,sync_dist=True)
        self.log("val_acc", self.valid_accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True,sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), self.hparams.lr, weight_decay=self.hparams.wd)

        # def lr_scheduler(epoch: int):
        #     if epoch <= self.hparams.lr_warmup_epochs:
        #         lr_scale = epoch / (self.hparams.lr_warmup_epochs - 1)
        #     else:
        #         lr_scale = (epoch - self.hparams.lr_warmup_epochs) / (
        #                 self.hparams.n_epochs - self.hparams.lr_warmup_epochs)
        #     return lr_scale
        #
        # lambda_lr = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_scheduler)
        # linear_warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-4, end_factor=1,
        #                                             total_iters=self.hparams.lr_warmup_epochs)
        linear_decay = optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=1e-4,
                                                   total_iters=self.hparams.n_epochs - self.hparams.lr_warmup_epochs)

        # scheduler = optim.lr_scheduler.SequentialLR(optimizer, [linear_warmup, linear_decay],
        #                                             milestones=[self.hparams.lr_warmup_epochs])
        # scheduler = create_lr_scheduler_with_warmup(linear_decay, 1e-6, self.hparams.lr_warmup_epochs, self.hparams.lr)

        return [optimizer], [linear_decay]
