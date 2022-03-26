"""
The main train file
    :author: gsc2001
    :brief: File to train on ImageNet
"""
import torch
import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pl_bolts.datamodules import ImagenetDataModule
from models.densenet import DensenetModule


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", help="Data dir for imagenet", required=True)
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    densenet = DensenetModule(32, (6, 12, 24, 16), 64)

    logger = WandbLogger(project="ConvexNets")
    trainer = pl.Trainer(max_epochs=90, logger=logger, gpus=torch.cuda.device_count())
    imagenet = ImagenetDataModule(args.data_dir)
    trainer.fit(densenet, imagenet)


if __name__ == '__main__':
    main()
