import os

import torch.utils.data
import torchvision.datasets
from enum import Enum
from .cifair import ciFAIR10, ciFAIR100

root_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')


class DatasetNames(Enum):
    STL10 = 'STL10'
    SVHN = 'SVHN'
    MNIST = 'MNIST'
    FMNIST = 'FMNIST'
    CIFAIR100 = 'CIFAIR100'
    CIFAIR10 = 'CIFAIR10'
    CIFAR100 = 'CIFAR100'
    CIFAR10 = 'CIFAR10'


class DatasetWrapper:
    def __init__(self, name: DatasetNames, train_set, test_set,
                 valid_frac=None):
        self.name = name
        self.test_set = test_set

        self.has_valid = valid_frac is not None
        valid_size = int(len(train_set) * valid_frac)
        if self.has_valid:
            indices = torch.randperm(len(train_set))
            train_indices = indices[:len(indices) - valid_size]
            valid_indices = indices[len(indices) - valid_size:]
            self.train_set = torch.utils.data.Subset(train_set, train_indices)
            self.valid_set = torch.utils.data.Subset(train_set, valid_indices)
        else:
            self.train_set = train_set
            self.valid_set = None


# class Transformer:
#     def __init__(self, transform=None, target_transform=None):
#         self.transform = transforms.Compose([*common_data_trans, *transform])
#         self.target_transform = transforms.Compose(
#             [*common_target_trans, *target_transform])
#
#
# transformers = {
#     DatasetNames.STL10: Transformer([], []),
#     DatasetNames.SVHN: Transformer([], []),
#     DatasetNames.CIFAR10: Transformer([], []),
#     DatasetNames.CIFAR100: Transformer([], []),
#     DatasetNames.MNIST: Transformer([], []),
#     DatasetNames.FMNIST: Transformer([], []),
#     DatasetNames.CIFAIR10: Transformer([], []),
#     DatasetNames.CIFAIR100: Transformer([], []),
# }

datasets = {
}


def get_dataset(name: DatasetNames, valid_frac=None, transform=None,
                target_transform=None, test_transform=None, **kwargs):
    if test_transform is None:
        test_transform = transform
    if name == DatasetNames.CIFAR10:
        train_set = torchvision.datasets.CIFAR10(
            root=os.path.join(root_dir, 'cifar10'), download=True, train=True,
            transform=transform, target_transform=target_transform)
        test_set = torchvision.datasets.CIFAR10(
            root=os.path.join(root_dir, 'cifar10'), download=True, train=False,
            transform=test_transform, target_transform=target_transform)
        return DatasetWrapper(name=name, train_set=train_set, test_set=test_set,
                              valid_frac=valid_frac)

    if name == DatasetNames.CIFAR100:
        train_set = torchvision.datasets.CIFAR100(
            root=os.path.join(root_dir, 'cifar100'), download=True, train=True,
            transform=transform, target_transform=target_transform)
        test_set = torchvision.datasets.CIFAR100(
            root=os.path.join(root_dir, 'cifar100'), download=True, train=False,
            transform=test_transform, target_transform=target_transform)
        return DatasetWrapper(name=name, train_set=train_set, test_set=test_set,
                              valid_frac=valid_frac)

    if name == DatasetNames.STL10:
        train_set = torchvision.datasets.STL10(
            root=os.path.join(root_dir, 'stl10'), download=True, split='train',
            transform=transform, target_transform=target_transform, **kwargs)
        test_set = torchvision.datasets.STL10(
            root=os.path.join(root_dir, 'stl10'), download=True, split='test',
            transform=test_transform, target_transform=target_transform, **kwargs)
        return DatasetWrapper(name=name, train_set=train_set, test_set=test_set,
                              valid_frac=valid_frac)

    if name == DatasetNames.FMNIST:
        train_set = torchvision.datasets.FashionMNIST(
            root=os.path.join(root_dir, 'fmnist'), download=True, train=True,
            transform=transform, target_transform=target_transform)
        test_set = torchvision.datasets.FashionMNIST(
            root=os.path.join(root_dir, 'fmnist'), download=True, train=False,
            transform=test_transform, target_transform=target_transform)
        return DatasetWrapper(name=name, train_set=train_set, test_set=test_set,
                              valid_frac=valid_frac)

    if name == DatasetNames.MNIST:
        train_set = torchvision.datasets.MNIST(
            root=os.path.join(root_dir, 'mnist'), download=True, train=True,
            transform=transform, target_transform=target_transform)
        test_set = torchvision.datasets.MNIST(
            root=os.path.join(root_dir, 'mnist'), download=True, train=False,
            transform=test_transform, target_transform=target_transform)
        return DatasetWrapper(name=name, train_set=train_set, test_set=test_set,
                              valid_frac=valid_frac)

    if name == DatasetNames.SVHN:
        train_set = torchvision.datasets.SVHN(
            root=os.path.join(root_dir, 'svhn'), download=True, split='train',
            transform=transform, target_transform=target_transform)
        test_set = torchvision.datasets.SVHN(
            root=os.path.join(root_dir, 'svhn'), download=True, split='test',
            transform=test_transform, target_transform=target_transform)
        return DatasetWrapper(name=name, train_set=train_set, test_set=test_set,
                              valid_frac=valid_frac)

    if name == DatasetNames.CIFAIR10:
        train_set = ciFAIR10(
            root=os.path.join(root_dir, 'cifair10'), download=True, train=True,
            transform=transform, target_transform=target_transform)
        test_set = ciFAIR10(
            root=os.path.join(root_dir, 'cifair10'), download=True, train=False,
            transform=test_transform, target_transform=target_transform)
        return DatasetWrapper(name=name, train_set=train_set, test_set=test_set,
                              valid_frac=valid_frac)

    if name == DatasetNames.CIFAIR100:
        train_set = ciFAIR100(
            root=os.path.join(root_dir, 'cifair100'), download=True, train=True,
            transform=transform, target_transform=target_transform)
        test_set = ciFAIR100(
            root=os.path.join(root_dir, 'cifair100'), download=True,
            train=False,
            transform=test_transform, target_transform=target_transform)
        return DatasetWrapper(name=name, train_set=train_set, test_set=test_set,
                              valid_frac=valid_frac)
