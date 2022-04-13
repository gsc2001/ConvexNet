from torch import nn
from tqdm import tqdm


def fit(model: nn.Module, train_loader, val_loader, optimizer, criterion, epochs, lr_scheduler=None):
    print('Starting fitting for', model.__class__.__name__)

    for epoch in range(epochs):
        # train epoch
        # test epochs
        # log results
        # step lr_scheduler
        pass



