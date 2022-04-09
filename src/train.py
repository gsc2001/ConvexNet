"""
The main train file
    :author: gsc2001
    :brief: File to train on ImageNet
"""
import os
import argparse

import torch
import pytorch_lightning as pl

import wandb

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from data_modules.imagenet import ImagenetDataModule
from models.densenet import DensenetModule
from models.ioc_densenet import IOCDensenetModule


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", help="Data dir for imagenet", required=True)
    parser.add_argument("--batch-size", help="Training batch_size", default=64, type=int)
    parser.add_argument("--convex", action='store_true')
    parser.add_argument("--save_dir", required=True)
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    wandb.login()
    dir_path = args.save_dir
    checkpoint_callback = ModelCheckpoint(
        monitor='val/acc',
        dirpath=dir_path,
        filename="imgnetDensenet-{epoch:02d}-{val/acc:.2f}",
        save_top_k=3,
        mode="max"
    )

    if args.convex:
        densenet = IOCDensenetModule(32, (6, 12, 24, 16), 64)
    else:
        densenet = DensenetModule(32, (6, 12, 24, 16), 64)

    logger = WandbLogger(project="ConvexNets")
    trainer = pl.Trainer(max_epochs=30, logger=logger, gpus=1, callbacks=[checkpoint_callback])
    imagenet = ImagenetDataModule(args.data_dir, batch_size=args.batch_size)
    trainer.fit(densenet, imagenet)


if __name__ == '__main__':
    main()
