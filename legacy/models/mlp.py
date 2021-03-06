import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from torch import nn

from tqdm import tqdm
import numpy as np

from datasets.preprocess import DatasetWrapper
from utils import AverageMeter


class MLP(torch.nn.Module):
    def __init__(self, input_features, out_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_features, 800),
            nn.BatchNorm1d(800),
            nn.ReLU(),
            nn.Linear(800, 800),
            nn.BatchNorm1d(800),
            nn.ReLU(),
            nn.Linear(800, 800),
            nn.BatchNorm1d(800),
            nn.ReLU(),
            nn.Linear(800, out_classes),
        )

    def forward(self, x):
        output = self.model(x)
        return output


def train_epoch(model, optimizer, loss_func, dataset, train_loader, epoch,
                n_epochs):
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

            batch_size = targets.size(0)
            _, pred = outputs.data.cpu().topk(1, dim=1)
            error = torch.ne(pred.squeeze(),
                             targets.cpu()).float().sum().item() / batch_size
            errors.update(error, batch_size)
            losses.update(loss.item())

            pbar.update(data.shape[0])
            pbar.set_postfix(**{
                '[Train/Loss]': losses.avg,
                '[Train/Error]': errors.avg
            })

    return losses.avg, errors.avg


def test_epoch(model: nn.Module, dataset: DatasetWrapper,
               test_loader: torch.utils.data.DataLoader):
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
                # loss = loss_func(outputs, targets)
                batch_size = targets.size(0)
                _, pred = outputs.data.cpu().topk(1, dim=1)
                error = torch.ne(pred.squeeze(),
                                 targets.cpu()).float().sum().item() / batch_size
                errors.update(error, batch_size)
                # losses.update(loss.item())

                pbar.update(data.shape[0])
                pbar.set_postfix(**{
                    # '[Valid/Loss]': losses.avg,
                    '[Valid/Error]': errors.avg
                })

    return None, errors.avg


def fit(model: MLP, dataset: DatasetWrapper, lr=0.0001, batch_size=64,
        n_epochs=10, path=None):
    if path is None:
        path = f'trained_models/mlp.{dataset.name}.random100'
    writer = SummaryWriter(f'runs/mlp.{dataset.name}.random100')
    if torch.cuda.is_available():
        model.cuda()
    model.train()
    train_loader = torch.utils.data.DataLoader(dataset.train_set,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=torch.cuda.is_available(),
                                               )
    print(train_loader.pin_memory)
    test_loader = torch.utils.data.DataLoader(dataset.test_set,
                                              batch_size=batch_size,
                                              pin_memory=torch.cuda.is_available(),
                                              )
    valid_loader = torch.utils.data.DataLoader(dataset.valid_set,
                                               batch_size=batch_size,
                                               pin_memory=torch.cuda.is_available(),
                                               )
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_error = np.inf
    counter = 0
    for epoch in range(n_epochs):
        train_loss, train_error = train_epoch(model=model, optimizer=optimizer,
                                              loss_func=loss_func,
                                              dataset=dataset,
                                              train_loader=train_loader,
                                              epoch=epoch,
                                              n_epochs=n_epochs)

        valid_loss, valid_error = test_epoch(model, dataset, valid_loader)
        writer.add_scalars('loss', {'train': train_loss}, epoch)
        writer.add_scalars('accuracy', {'train': (1 - train_error) * 100,
                                        'valid': (1 - valid_error) * 100},
                           epoch)

        print(valid_error)

        if valid_error < best_error:
            print('Saving!')
            torch.save(model.state_dict(), path)
            best_error = valid_error
            counter = 0
        else:
            counter += 1

        if counter > 7:
            print("Patience came ending now")
            break

# current_loss = 0.0
