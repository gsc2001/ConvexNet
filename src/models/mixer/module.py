import pytorch_lightning as pl
from torch import optim
import torch.nn.functional as F
from .model import MlpMixer
from utils import mixup_data, mixup_criterion


class MixerModule(pl.LightningModule):
    def __init__(self, model, lr=1e-3, wd=.1, lr_warmup_epochs=35, n_epochs=300, mixup_p=.5):
        super().__init__()
        self.save_hyperparameters()
        self.model = model

    def forward(self, x):
        y = self.model(x)
        return y

    def training_step(self, batch, batch_idx):
        x, y = batch
        x, y_a, y_b, lam = mixup_data(x, y, self.hparams.mixup_p)
        y_hat = self(x)
        loss = mixup_criterion(F.cross_entropy, y_hat, y_a, y_b, lam)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), self.hparams.lr, weight_decay=self.hparams.wd)

        def lr_scheduler(epoch: int):
            if epoch <= self.hparams.lr_warmpup_epochs:
                lr_scale = epoch / (self.hparams.lr_warmup_epochs - 1)
            else:
                lr_scale = (epoch - self.hparams.lr_warmup_epochs) / (
                        self.hparams.n_epochs - self.hparams.lr_warmup_epochs)
            return lr_scale

        lambda_lr = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_scheduler)

        return [optimizer], [lambda_lr]
