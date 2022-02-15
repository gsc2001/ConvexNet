#!/usr/bin/env python

from datasets.preprocess import get_dataset, DatasetNames
from timm.data.transforms_factory import create_transform
from timm.data import create_dataset
from timm.data.loader import create_loader

from torchvision.transforms import transforms
from models import mixer
from config import Mixer_B_16_config as cfg
import pytorch_lightning as pl
import sys


# hyper parameters


def main():
    cfg.num_classes = 1000
    imnet_transforms = create_transform(cfg.img_size, is_training=True,
                                        auto_augment=f'rand-m{cfg.rand_aug_magnitude}-n{cfg.rand_aug_num_ops}')

    train_dataset = create_dataset('tfds/imagenet2012', root=sys.argv[1], batch_size=cfg.batch_size,
                                   is_training=True, split='train')

    train_loader = create_loader(train_dataset, cfg.img_size, cfg.batch_size,
                                 is_training=True,
                                 auto_augment=f'rand-m{cfg.rand_aug_magnitude}-n{cfg.rand_aug_num_ops}')
    valid_dataset = create_dataset('tfds/imagenet2012', root=sys.argv[1], batch_size=cfg.batch_size, split='validation')
    valid_loader = create_loader(valid_dataset, cfg.img_size, cfg.batch_size)

    model = mixer.MlpMixer(cfg)
    module = mixer.MixerModule(model, cfg.lr, cfg.weight_decay, cfg.lr_warmup_epochs, cfg.num_epochs,
                               cfg.mixup_strength)
    trainer = pl.Trainer(gradient_clip_val=1, gpus=[0])
    trainer.fit(module, train_loader, valid_loader)

    # ioc_mixer.fit(model, dataset, batch_size=BATCH_SIZE, n_epochs=EPOCHS)


if __name__ == '__main__':
    main()
