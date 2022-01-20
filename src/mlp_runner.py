import torch
from torchvision.transforms import transforms
from models import mlp, ioc_mlp
from datasets.preprocess import DatasetWrapper, DatasetNames, get_dataset

classes = {
    DatasetNames.STL10: 10,
    DatasetNames.SVHN: 10,
    DatasetNames.CIFAR10: 10,
    DatasetNames.CIFAR100: 100,
    DatasetNames.MNIST: 10,
    DatasetNames.FMNIST: 10,
    DatasetNames.CIFAIR10: 10,
    DatasetNames.CIFAIR100: 100,
}

in_features = {
    DatasetNames.STL10: 3 * 96 * 96,
    DatasetNames.SVHN: 3 * 32 * 32,
    DatasetNames.CIFAR10: 3 * 32 * 32,
    DatasetNames.CIFAR100: 3 * 32 * 32,
    DatasetNames.MNIST: 28 * 28,
    DatasetNames.FMNIST: 28 * 28,
    DatasetNames.CIFAIR10: 3 * 32 * 32,
    DatasetNames.CIFAIR100: 3 * 32 * 32,
}


def main():
    print('Running MLP')
    for dataset_name in DatasetNames:
        dataset = get_dataset(dataset_name, valid_frac=.2,
                              transform=transforms.Compose(
                                  [transforms.ToTensor()]))

        model = mlp.MLP(in_features[dataset_name], classes[dataset_name])
        mlp.fit(model, dataset)

    print('Running IOC-MLP')
    for dataset_name in DatasetNames:
        dataset = get_dataset(dataset_name, valid_frac=.2,
                              transform=transforms.Compose(
                                  [transforms.ToTensor()]))

        model = ioc_mlp.IOC_MLP(in_features[dataset_name],
                                classes[dataset_name])
        ioc_mlp.fit(model, dataset)

if __name__ == '__main__':
    main()