#!/usr/bin/env python

from datasets.preprocess import get_dataset, DatasetNames
from timm.data.transforms_factory import create_transform
from timm.data import create_dataset
from timm.data.loader import create_loader

from torchvision.transforms import transforms
from torchvision.datasets import ImageNet
from torch.utils.data import DataLoader
from models import mixer
from config import Mixer_B_16_config as cfg
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import sys


# hyper parameters


def main():
    cfg.num_classes = 1000
    print(cfg.img_size, cfg.patch_size)
    imnet_transforms = create_transform(cfg.img_size, is_training=True,
                                        auto_augment=f'rand-m{cfg.rand_aug_magnitude}-n{cfg.rand_aug_num_ops}')
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        dirpath='imgnet_models/mlpMixer/',
        filename="imgnetMixer-{epoch:02d}-{val_acc:.2f}",
        save_top_k=3,
        mode="max"
    )
    train_dataset = ImageNet(sys.argv[1], split='train', transform=imnet_transforms, shuffle=True)
    valid_dataset = ImageNet(sys.argv[1], split='validation', transform=create_transform(cfg.img_size))
    # train_dataset = create_dataset('tfds/imagenet2012', root=sys.argv[1], batch_size=cfg.batch_size,
    #                                is_training=True, split='train', transform=imnet_transforms)

    # valid_dataset = create_dataset('tfds/imagenet2012', root=sys.argv[1], batch_size=cfg.batch_size, split='validation',
    #                                download=True, transform=create_transform(224))
    train_loader = DataLoader(train_dataset, cfg.batch_size, num_workers=10, shuffle=True)
    valid_loader = DataLoader(valid_dataset, cfg.batch_size, num_workers=10)

    model = mixer.MlpMixer(cfg)
    module = mixer.MixerModule(model, cfg.lr, cfg.weight_decay, cfg.lr_warmup_epochs, cfg.num_epochs,
                               cfg.mixup_strength)
    trainer = pl.Trainer(gradient_clip_val=1, gpus=[0], callbacks=[checkpoint_callback])
    trainer.fit(module, train_loader, valid_loader)

    # ioc_mixer.fit(model, dataset, batch_size=BATCH_SIZE, n_epochs=EPOCHS)


if __name__ == '__main__':
    main()
