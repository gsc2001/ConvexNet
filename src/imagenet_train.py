#!/usr/bin/env python

from datasets.preprocess import get_dataset, DatasetNames
from timm.data.transforms_factory import create_transform

from torchvision.transforms import transforms
from models import ioc_mixer
from config import Mixer_B_16_config


# hyper parameters


def main():
    Mixer_B_16_config.num_classes = 1000
    imnet_transforms = create_transform(Mixer_B_16_config.img_size, is_training=True,
                                        auto_augment=f'rand-m{Mixer_B_16_config.rand_aug_magnitude}-n{Mixer_B_16_config.rand_aug_num_ops}')

    dataset = get_dataset(DatasetNames.CIFAR10, valid_frac=0.2, transform=imnet_transforms)

    # model = ioc_mixer.MlpMixer(3, Mixer_B_16_config)

    # ioc_mixer.fit(model, dataset, batch_size=BATCH_SIZE, n_epochs=EPOCHS)


if __name__ == '__main__':
    main()
