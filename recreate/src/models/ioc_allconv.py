import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from torch import nn
import torch.nn.functional as F

from tqdm import tqdm
import numpy as np

from datasets.preprocess import DatasetWrapper
from utils import AverageMeter


class IOC_AllConv(nn.Module):
    def __init__(self, input_size, n_classes=10, **kwargs):
        super(IOC_AllConv, self).__init__()
        self.conv1 = nn.Conv2d(input_size, 96, 3, padding=1)
        self.conv2 = nn.Conv2d(96, 96, 3, padding=1)
        self.conv3 = nn.Conv2d(96, 96, 3, padding=1, stride=2)
        self.conv4 = nn.Conv2d(96, 192, 3, padding=1)
        self.conv5 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv6 = nn.Conv2d(192, 192, 3, padding=1, stride=2)
        self.conv7 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv8 = nn.Conv2d(192, 192, 1)

        self.class_conv = nn.Conv2d(192, n_classes, 1)

    def forward(self, x):
        x_drop = F.dropout(x, .2)
        conv1_out = F.elu(self.conv1(x_drop))
        conv2_out = F.elu(self.conv2(conv1_out))
        conv3_out = F.elu(self.conv3(conv2_out))
        conv3_out_drop = F.dropout(conv3_out, .5)
        conv4_out = F.elu(self.conv4(conv3_out_drop))
        conv5_out = F.elu(self.conv5(conv4_out))
        conv6_out = F.elu(self.conv6(conv5_out))
        conv6_out_drop = F.dropout(conv6_out, .5)
        conv7_out = F.elu(self.conv7(conv6_out_drop))
        conv8_out = F.elu(self.conv8(conv7_out))

        class_out = F.elu(self.class_conv(conv8_out))
        pool_out = F.adaptive_avg_pool2d(class_out, 1)
        pool_out.squeeze_(-1)
        pool_out.squeeze_(-1)
        return pool_out


def train_epoch(model: nn.Module, optimizer, loss_func, dataset, train_loader,
                epoch,
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

            # convex ensuring step:
            for name, param in model.named_parameters():
                if 'weight' in name and 'conv1' not in name:
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

            pbar.update(data.size(0))
            pbar.set_postfix(**{
                '[Train/Loss]': losses.avg,
                '[Train/Error]': errors.avg
            })

    return losses.avg, errors.avg


#
#
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
                    '[Valid/Error]': errors.avg
                })

    return errors.avg


def fit(model: IOC_AllConv, dataset: DatasetWrapper, lr=0.001, batch_size=64,
        n_epochs=10, path=None):
    if path is None:
        path = f'trained_models/ioc_allconv.{dataset.name}'
    print('hi')
    writer = SummaryWriter(f'runs/ioc_allconv.{dataset.name}')
    if torch.cuda.is_available():
        model.cuda()
    model.train()
    train_loader = torch.utils.data.DataLoader(dataset.train_set,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               )
    test_loader = torch.utils.data.DataLoader(dataset.test_set,
                                              batch_size=batch_size,
                                              )
    valid_loader = torch.utils.data.DataLoader(dataset.valid_set,
                                               batch_size=batch_size,
                                               )
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=.9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(10 * 2 / 3.5), int(10 * 2.5 / 3.5), int(10 * 3 / 3.5)],
        gamma=0.1)

    best_error = np.inf
    counter = 0
    for epoch in range(n_epochs):
        train_loss, train_error = train_epoch(model=model, optimizer=optimizer,
                                              loss_func=loss_func,
                                              dataset=dataset,
                                              train_loader=train_loader,
                                              epoch=epoch,
                                              n_epochs=n_epochs)

        valid_error = test_epoch(model, dataset, valid_loader)
        writer.add_scalars('loss', {'train': train_loss}, epoch)
        writer.add_scalars('accuracy', {'train': (1 - train_error) * 100,
                                        'valid': (1 - valid_error) * 100},
                           epoch)
        print(valid_error)
        #
        scheduler.step()

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
    writer.close()
