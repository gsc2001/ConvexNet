"""
The main train file
    :author: gsc2001
    :brief: File to train on ImageNet
"""
import torch
import argparse
import pytorch_lightning as pl

import wandb

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from data_modules.imagenet import ImagenetDataModule
from models.densenet import DensenetModule


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", help="Data dir for imagenet", required=True)
    parser.add_argument("--batch-size", help="Training batch_size", default=64, type=int)
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    wandb.login()

    checkpoint_callback = ModelCheckpoint(
        monitor='val/acc',
        dirpath='imgnet_models/densenet/',
        filename="imgnetMixer-{epoch:02d}-{val/acc:.2f}",
        save_top_k=3,
        mode="max"
    )

    densenet = DensenetModule(32, (6, 12, 24, 16), 64)

    logger = WandbLogger(project="ConvexNets")
    trainer = pl.Trainer(max_epochs=90, logger=logger, gpus=torch.cuda.device_count(), callbacks=[checkpoint_callback])
    imagenet = ImagenetDataModule(args.data_dir, batch_size=args.batch_size)
    trainer.fit(densenet, imagenet)


if __name__ == '__main__':
    main()
