import os
import time
import torch
import torch.utils.data
import torch.nn.functional
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from tqdm import tqdm

from .densenet import IOC_DenseNet
from datasets.preprocess import DatasetWrapper


class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_epoch(model, loader, dataset: DatasetWrapper, optimizer, epoch,
                n_epochs, print_freq=1):
    batch_time = AverageMeter()
    losses = AverageMeter()
    error = AverageMeter()

    # Model on train mode
    model.train()

    end = time.time()

    with tqdm(total=len(dataset.train_set),
              desc=f'Epoch {epoch + 1} / {n_epochs}') as pbar:
        for batch_idx, (input, target) in enumerate(loader):
            # Create vaiables
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            # compute output
            output = model(input)
            loss = torch.nn.functional.cross_entropy(output, target)

            # measure accuracy and record loss
            batch_size = target.size(0)
            _, pred = output.data.cpu().topk(1, dim=1)
            error.update(torch.ne(pred.squeeze(),
                                  target.cpu()).float().sum().item() / batch_size,
                         batch_size)
            losses.update(loss.item(), batch_size)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for name, param in model.named_parameters():
                if ('weight' in name) and ('conv0' not in name):
                    param_data = param.data.cpu().numpy()
                    param_data[param_data < 0] = np.exp(
                        param_data[param_data < 0] - 5)
                    param.data.copy_(torch.tensor(param_data))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            pbar.update(input.size(0))
            pbar.set_postfix(**{
                '[Train/Loss]': losses.avg,
                '[Train/Error]': error.avg
            })

    # Return summary statistics
    return batch_time.avg, losses.avg, error.avg


def test_epoch(model, dataset: DatasetWrapper, loader):
    batch_time = AverageMeter()
    losses = AverageMeter()
    error = AverageMeter()

    # Model on eval mode
    model.eval()

    end = time.time()
    with tqdm(total=len(dataset.test_set), desc='Valid') as pbar:
        with torch.no_grad():
            for batch_idx, (input, target) in enumerate(loader):
                # Create vaiables
                if torch.cuda.is_available():
                    input = input.cuda()
                    target = target.cuda()

                # compute output
                output = model(input)
                loss = torch.nn.functional.cross_entropy(output, target)

                # measure accuracy and record loss
                batch_size = target.size(0)
                _, pred = output.data.cpu().topk(1, dim=1)
                error.update(torch.ne(pred.squeeze(),
                                      target.cpu()).float().sum().item() / batch_size,
                             batch_size)
                losses.update(loss.item(), batch_size)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # print stats
                pbar.update(input.shape[0])
                pbar.set_postfix(**{
                    '[Valid/Error]': error.avg
                })

    # Return summary statistics
    return batch_time.avg, losses.avg, error.avg


def fit(model, dataset: DatasetWrapper, n_epochs=10, batch_size=64, lr=0.1,
        wd=0.0001, momentum=0.9, path=None):
    # if seed is not None:
    #     torch.manual_seed(seed)
    if path is None:
        path = f'trained_models/densenet.{dataset.name}'

    # Data loaders
    train_loader = torch.utils.data.DataLoader(dataset.train_set,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=(
                                                   torch.cuda.is_available()),
                                               num_workers=0)
    test_loader = torch.utils.data.DataLoader(dataset.test_set,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=(
                                                  torch.cuda.is_available()),
                                              num_workers=0)
    valid_loader = torch.utils.data.DataLoader(dataset.valid_set,
                                               batch_size=batch_size,
                                               shuffle=False, pin_memory=(
            torch.cuda.is_available()), num_workers=0)
    # Model on cuda
    if torch.cuda.is_available():
        model = model.cuda()

    # Wrap model for multi-GPUs, if necessary
    model_wrapper = model
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model_wrapper = torch.nn.DataParallel(model).cuda()

    # Optimizer
    optimizer = torch.optim.SGD(model_wrapper.parameters(), lr=lr,
                                momentum=momentum, nesterov=True,
                                weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=[
                                                         int(0.5 * n_epochs),
                                                         int(0.75 * n_epochs)],
                                                     gamma=0.1)

    # Start log

    writer = SummaryWriter(f'runs/densenet.{dataset.name}')
    # Train model
    best_error = 1
    counter = 0
    for epoch in range(n_epochs):
        _, train_loss, train_error = train_epoch(
            model=model_wrapper,
            loader=train_loader,
            dataset=dataset,
            optimizer=optimizer,
            epoch=epoch,
            n_epochs=n_epochs,
        )
        scheduler.step()
        _, valid_loss, valid_error = test_epoch(
            model=model_wrapper,
            dataset=dataset, loader=valid_loader)

        writer.add_scalars('loss', {'train': train_loss}, epoch)
        writer.add_scalars('accuracy', {'train': (1 - train_error) * 100,
                                        'valid': (1 - valid_error) * 100},
                           epoch)

        print(valid_error)

        # Determine if model is the best
        if valid_error < best_error:
            print('Saving')
            best_error = valid_error
            torch.save(model.state_dict(), path)
            counter = 0
        else:
            counter += 1

        if counter > 7:
            print('Patience came ending now!')

        writer.close()

    # Final test of model on test set

# def fit(model, dataset: DatasetWrapper, save, depth=100, growth_rate=12,
#          efficient=True,
#          valid_size=5000,
#          n_epochs=300, batch_size=64, seed=None):
#     """
#     A demo to show off training of efficient DenseNets.
#     Trains and evaluates a DenseNet-BC on CIFAR-10.
#
#     Args:
#         data (str) - path to directory where data should be loaded from/downloaded
#             (default $DATA_DIR)
#         save (str) - path to save the model to (default /tmp)
#
#         depth (int) - depth of the network (number of convolution layers) (default 40)
#         growth_rate (int) - number of features added per DenseNet layer (default 12)
#         efficient (bool) - use the memory efficient implementation? (default True)
#
#         valid_size (int) - size of validation set
#         n_epochs (int) - number of epochs for training (default 300)
#         batch_size (int) - size of minibatch (default 256)
#         seed (int) - manually set the random seed (default None)
#     """
#
#     # Get densenet configuration
#     if (depth - 4) % 3:
#         raise Exception('Invalid depth')
#     block_config = [(depth - 4) // 6 for _ in range(3)]
#
#     # Data transforms
#     # mean = [0.49139968, 0.48215841, 0.44653091]
#     # stdv = [0.24703223, 0.24348513, 0.26158784]
#     # train_transforms = transforms.Compose([
#     #     transforms.RandomCrop(32, padding=4),
#     #     transforms.RandomHorizontalFlip(),
#     #     transforms.ToTensor(),
#     #     transforms.Normalize(mean=mean, std=stdv),
#     # ])
#     # test_transforms = transforms.Compose([
#     #     transforms.ToTensor(),
#     #     transforms.Normalize(mean=mean, std=stdv),
#     # ])
#     #
#     # # Datasets
#     # train_set = datasets.CIFAR10(data, train=True, transform=train_transforms,
#     #                              download=True)
#     # test_set = datasets.CIFAR10(data, train=False, transform=test_transforms,
#     #                             download=False)
#
#     if valid_size:
#         valid_set = datasets.CIFAR10(data, train=True,
#                                      transform=test_transforms)
#         indices = torch.randperm(len(train_set))
#         train_indices = indices[:len(indices) - valid_size]
#         valid_indices = indices[len(indices) - valid_size:]
#         train_set = torch.utils.data.Subset(train_set, train_indices)
#         valid_set = torch.utils.data.Subset(valid_set, valid_indices)
#     else:
#         valid_set = None
#
#     # Models
#     model = DenseNet(
#         growth_rate=growth_rate,
#         block_config=block_config,
#         num_init_features=growth_rate * 2,
#         num_classes=10,
#         small_inputs=True,
#         efficient=efficient,
#     )
#     print(model)
#
#     # Print number of parameters
#     num_params = sum(p.numel() for p in model.parameters())
#     print("Total parameters: ", num_params)
#
#     # Make save directory
#     if not os.path.exists(save):
#         os.makedirs(save)
#     if not os.path.isdir(save):
#         raise Exception('%s is not a dir' % save)
#
#     # Train the model
#     train(model=model, train_set=train_set, valid_set=valid_set,
#           test_set=test_set, save=save,
#           n_epochs=n_epochs, batch_size=batch_size, seed=seed)
#     print('Done!')
