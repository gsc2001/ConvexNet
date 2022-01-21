import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from tqdm import tqdm
import numpy as np

from .model import MlpMixer
from datasets.preprocess import DatasetWrapper
from utils import get_loaders, AverageMeter


def train_epoch(model, optimizer, loss_func, dataset, train_loader, epoch,
                n_epochs, train_epoch_hook=None):
    model.train()
    losses = AverageMeter()
    errors = AverageMeter()
    with tqdm(total=len(dataset.train_set),
              desc=f"Epoch {epoch + 1} / {n_epochs}") as pbar:
        for data, targets in train_loader:
            if torch.cuda.is_available():
                data = data.cuda()
                targets = targets.cuda()

            optimizer.zero_grad()
            outputs = model(data)

            loss = loss_func(outputs, targets)

            loss.backward()

            optimizer.step()

            # convex ensuring step:
            for name, param in model.named_parameters():
                if 'weight' in name:
                    if 'to_patch_embedding' in name:
                        continue
                    if 'token_mix.0' in name:
                        if 'mixer_blocks.0' in name:
                            continue
                    param_data = param.data.cpu().numpy()
                    param_data[param_data < 0] = np.exp(
                        param_data[param_data < 0] - 5)
                    #
                    param.data.copy_(torch.tensor(param_data))

            batch_size = targets.size(0)
            _, pred = outputs.data.cpu().topk(1, dim=1)
            error = torch.ne(pred.squeeze(),
                             targets.cpu()).float().sum().item() / batch_size
            errors.update(error, batch_size)
            losses.update(loss.item())

            if train_epoch_hook:
                train_epoch_hook(outputs, targets, loss)

            pbar.update(data.shape[0])
            pbar.set_postfix(**{
                '[Train/Loss]': losses.avg,
                '[Train/Error]': errors.avg
            })

    return losses.avg, errors.avg


def test_epoch(model: nn.Module, dataset: DatasetWrapper, loss_func,
               test_loader: torch.utils.data.DataLoader, test_epoch_hook=None):
    model.eval()
    # losses = AverageMeter()
    errors = AverageMeter()

    with tqdm(total=len(dataset.test_set),
              desc=f"Valid") as pbar:
        with torch.no_grad():
            for data, targets in test_loader:
                if torch.cuda.is_available():
                    data = data.cuda()
                    targets = targets.cuda()
                outputs = model(data)
                loss = loss_func(outputs, targets)
                batch_size = targets.size(0)
                _, pred = outputs.data.cpu().topk(1, dim=1)
                error = torch.ne(pred.squeeze(),
                                 targets.cpu()).float().sum().item() / batch_size
                errors.update(error, batch_size)
                # losses.update(loss.item())

                if test_epoch_hook:
                    test_epoch_hook(outputs, targets, loss)

                pbar.update(data.shape[0])
                pbar.set_postfix(**{
                    # '[Valid/Loss]': losses.avg,
                    '[Valid/Error]': errors.avg
                })

    return errors.avg


def fit(model: MlpMixer, dataset: DatasetWrapper, lr=1e-3, batch_size=64,
        n_epochs=10, save_name=None, train_epoch_hook=None,
        valid_epoch_hook=None):
    if save_name is None:
        save_name = f'ioc_mlpmixer.{dataset.name}'

    writer = SummaryWriter(f'runs/{save_name}')

    if torch.cuda.is_available():
        model.cuda()

    model.train()

    loaders = get_loaders(dataset, batch_size)

    loss_func = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_error = np.inf
    counter = 0

    for epoch in range(n_epochs):
        train_loss, train_error = train_epoch(model, optimizer, loss_func,
                                              dataset, loadeclears['train'], epoch,
                                              n_epochs=n_epochs,
                                              train_epoch_hook=train_epoch_hook)

        valid_error = test_epoch(model, dataset, loss_func=loss_func,
                                 test_loader=loaders['valid'],
                                 test_epoch_hook=valid_epoch_hook)

        writer.add_scalars('loss', {'train': train_loss}, epoch)
        writer.add_scalars('accuracy', {'train': (1 - train_error) * 100,
                                        'valid': (1 - valid_error) * 100},
                           epoch)

        print(valid_error)

        if valid_error < best_error:
            print('Saving!')
            torch.save(model.state_dict(), f'trained_models/{save_name}')
            best_error = valid_error
        #     counter = 0
        # else:
        #     counter += 1

        # if counter > 7:
        #     print("Patience came ending now")
        #     break
