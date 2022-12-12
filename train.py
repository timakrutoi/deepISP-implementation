#!/usr/bin/python3
#encode=utf-8

import torch.optim as optim
from tqdm import tqdm

from network import DeepISP
from loss import deepISPloss
from dataset import get_data

# 
def train(
    epoch, start_epoch,
    batch_size, dataset_path, data_factor, crop_size, target,
    checkpoint_path, make_checkpoints,
    Nll, Nhl,
    lr, momentum):

    epochs = [i for i in range(start_epoch, epoch)]
    train, test = get_data(data_path=dataset_path, batch_size=batch_size, target=target, factor=data_factor, crop_size=crop_size)

    model = DeepISP(Nll, Nhl).float()
    criterion = deepISPloss()
    optimizer = optim.SGD(DeepISP.parameters(model), lr, momentum)

    test_loss = 0

    print(f'Params :')
    print(f'\tStart epoch: {start_epoch}, End epoch {epoch}')
    print(f'\tTrain set size: {len(train)}, Test set size: {len(test)}')
    print(f'\tLearning rate: {lr}, Momentum {momentum}')
    print(f'\tN lowlevel: {Nll}, N highlevel {Nhl}')
    print()
    print('Starting trainig...')

    for epoch in epochs:
        train_iter = tqdm(train, ncols=100, desc='Epoch: {}, training'.format(epoch))
        for (x, target) in train_iter:
            optimizer.zero_grad()
            y = model(x.float())
            loss = criterion(y, target)

    #         loss.backward()
            optimizer.step()

        test_iter = tqdm(test, ncols=128, desc='Epoch: {}, testing '.format(epoch))
        for idx, (x, target) in enumerate(test_iter):
            y = model(x.float())
            loss = criterion(y, target)
            test_loss += loss
            test_iter.set_postfix(str=f'loss: {test_loss / (idx + 1)}')
        test_loss /= len(test_iter)

        if make_checkpoints:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': test_loss,
            }, checkpoint_path + '/model_e{}_loss{}'.format(epoch, test_loss))

    print('Training done!')


if __name__ == '__main__':
    from argparse import ArgumentParser

    p = ArgumentParser()
    p.add_argument('-b', '--batch-size', type=int, default=1,
        help='Size of one training/testing batch. Default = %(default)d.')
    p.add_argument('--Nll', type=int, default=5,
        help='Number of lowlevel layers. Default = %(default)d.')
    p.add_argument('--Nhl', type=int, default=5,
        help='Number of highlevel layers. Default = %(default)d.')
    p.add_argument('--lr', type=float, default=0.1,
        help='Learning rate. Default = %(default)f.')
    p.add_argument('--momentum', type=float, default=0.9,
        help='Momentum. Default = %(default)f.')

    p.add_argument('-d', '--dataset-path', type=str, default='../dataset/S7-ISP-Dataset',
        help='Path to dataset. Default = %(default)s.')
    p.add_argument('-t', '--target', type=str, choices=['m', 'l'], default='m',
        help='Target mode. Default = %(default)s.')
    p.add_argument('--data-factor', type=float, default=0.7,
        help='Factor used to devide data between train and test sets. Default = %(default)f.')
    p.add_argument('--crop-size', type=int, default=256,
        help='Size of training pics. Default = %(default)dx%(default)d.')

    p.add_argument('--checkpoint-path', type=str, default='checkpoints',
        help='Path to checkpoint dir (used for both saving and loading). Default = %(default)s.')
    p.add_argument('-c', '--make-checkpoints', type=bool, default=True,
        help='Make checkpoints every epoch or not. Default = %(default)s.')

    p.add_argument('-e', '--epoch', type=int, default=10,
        help='Number of epochs to train. Default = %(default)d.')
    p.add_argument('--start-epoch', type=int, default=0,
        help='Starting epoch (loading checkpoint). Default = %(default)d.')

    args = p.parse_args()

    train(**vars(args))
