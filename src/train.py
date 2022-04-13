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
from torchvision import transforms
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from data_modules.imagenet import ImagenetDataModule
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from models.densenet import DensenetModule
from models.ioc_densenet import IOCDensenetModule


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
    parser = DensenetModule.add_model_specific_args(parser)

    args = parser.parse_args()

    wandb.login()
    dir_path = args.save_dir
    checkpoint_callback = ModelCheckpoint(
        monitor='val/acc',
        dirpath=dir_path,
        filename="imgnetDensenet-{epoch:02d}-{val/acc:.2f}",
        save_top_k=3,
        mode="max"
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    if args.convex:
        densenet = IOCDensenetModule(32, (6, 12, 24, 16), 64, lr=args.lr, epochs=args.epochs,
                                     num_classes=args.num_classes, drop_rate=args.drop_rate)
    else:
        densenet = DensenetModule(32, (6, 12, 24, 16), 64, lr=args.lr, epochs=args.epochs, num_classes=args.num_classes,
                                  drop_rate=args.drop_rate)

    logger = WandbLogger(project="ConvexNets")
    trainer = pl.Trainer(max_epochs=args.epochs, logger=logger, gpus=1, callbacks=[checkpoint_callback, lr_monitor])
    if args.dataset == 'imagenet':
        datamodule = ImagenetDataModule(args.data_dir, batch_size=args.batch_size)
    else:
        train_transforms = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            cifar10_normalization()
        ])
        val_transforms = transforms.Compose([
            transforms.ToTensor(),
            cifar10_normalization()
        ])
        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            cifar10_normalization()
        ])
        datamodule = CIFAR10DataModule(args.data_dir, batch_size=args.batch_size, train_transforms=train_transforms,
                                       val_transforms=val_transforms, test_transforms=test_transforms,
                                       num_workers=os.cpu_count() // 3)
    print('Dataset: ', datamodule.__class__.__name__)
    print('Model: ', densenet.__class__.__name__)
    print('lr: ', args.lr)

    trainer.fit(densenet, datamodule=datamodule)


if __name__ == '__main__':
    main()
