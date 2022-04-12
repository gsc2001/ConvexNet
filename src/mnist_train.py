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
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pl_bolts.datamodules import MNISTDataModule
from models.mnist_model import NNModule


def add_global_args(parent_parser: argparse.ArgumentParser):
    parser = parent_parser.add_argument_group('Global')
    parser.add_argument("--data-dir", help="Data dir for imagenet", required=True)
    parser.add_argument("--batch-size", help="Training batch_size", default=64, type=int)
    parser.add_argument("--convex", action='store_true')
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--dataset", default='imagenet')
    parser.add_argument("--epochs", type=int, default=30)
    return parent_parser


def main():
    parser = argparse.ArgumentParser()
    parser = add_global_args(parser)
    parser = NNModule.add_model_specific_args(parser)

    args = parser.parse_args()

    wandb.login()
    dir_path = args.save_dir
    # checkpoint_callback = ModelCheckpoint(
    #     monitor='val/acc',
    #     dirpath=dir_path,
    #     filename="imgnetDensenet-{epoch:02d}-{val/acc:.2f}",
    #     save_top_k=3,
    #     mode="max"
    # )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    model = NNModule(lr=args.lr)

    logger = WandbLogger(project="ConvexNets")
    trainer = pl.Trainer(max_epochs=args.epochs, logger=logger, gpus=1, callbacks=[lr_monitor])
    datamodule = MNISTDataModule(data_dir=args.data_dir, batch_size=args.batch_size)
    print('Dataset: ', datamodule.__class__.__name__)
    print('Model: ', model.__class__.__name__)
    print('lr: ', args.lr)

    trainer.fit(model, datamodule=datamodule)


if __name__ == '__main__':
    main()
