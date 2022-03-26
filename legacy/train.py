#!/usr/bin/env python

"""
    :brief: File to train specific model
    :author: gsc2001
"""
from datasets.preprocess import get_dataset, DatasetNames

from torchvision.transforms import transforms
from models import ioc_mixer

# hyper parameters

RESIZE_TO = 72
PATCH_SIZE = 9

NUM_MIXER_LAYERS = 4
HIDDEN_SIZE = 128
MLP_SEQ_DIM = 64
MLP_CHANNEL_DIM = 128

EPOCHS = 100
BATCH_SIZE = 512


def main():
    mean = [0.49139968, 0.48215841, 0.44653091]
    stdv = [0.24703223, 0.24348513, 0.26158784]
    dataset = get_dataset(DatasetNames.CIFAR10, valid_frac=.2,
                          transform=transforms.Compose(
                              [
                                  transforms.Resize(RESIZE_TO),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean, stdv)
                              ]))

    model = ioc_mixer.MlpMixer(3, HIDDEN_SIZE, 10, PATCH_SIZE, RESIZE_TO,
                               NUM_MIXER_LAYERS, MLP_SEQ_DIM, MLP_CHANNEL_DIM)

    ioc_mixer.fit(model, dataset, batch_size=BATCH_SIZE, n_epochs=EPOCHS)


if __name__ == '__main__':
    main()
