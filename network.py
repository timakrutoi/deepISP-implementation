#!/usr/bin/python3
#encode=utf-8

import torch
import torch.nn as nn

import numpy as np


class DeepispLL(nn.Module):
    def __init__(self, kernel=(3,3), stride=1):
        super(DeepispLL, self).__init__()

        # self.size = n * m
        self.kernel = kernel
        self.stride = stride

    def forward(self, x):
        # rsize = self.size * 61
        # lsize = self.size * 3
        rh = nn.Conv2d(61, 61, kernel=self.kernel, stride=self.stride)(x[:,:,:61])
        rh = nn.RelU()(rh)

        lh = nn.Conv2d(3, 3, kernel=self.kernel, stride=self.stride)(x[:,:,61:])
        lh = nn.Tanh()(lh)

        # need to so some sum
        # lh += x

        res = np.empty(x.shape)
        res[:, :, :61] = rh
        res[:, :, 61:] = lh

        return res


class DeepispHL(nn.Module):
    def __init__(self, kernel=(3,3), stride=2):
        super(DeepispHL, self).__init__()

        self.kernel = kernel
        self.stride = stride

    def forward(self, x):
        x = nn.Conv2d(64, 64, kernel=self.kernel, stride=self.stride)(x)
        x = nn.RelU()(x)
        x = nn.MaxPool2d(2, 2)

        return x


class DeepISP(nn.Module):
    def __init__(self, n_ll, n_hl):
        super(DeepISP, self).__init__()

        self.lowlevel = nn.Sequential()
        self.highlevel = nn.Sequential()

        for i in range(n_ll):
            self.lowlevel.append(DeepispLL())

        for i in range(n_hl):
            self.highlevel.append(DeepispHL())

        # append global pooling to high level to get 1x1x64 shape
        # current shape = (N/4^n_hl)x(M/4^n_hl)x64
        # self.highlevel.append(nn.MaxPool2(...))

        self.highlevel.append(nn.Linear(64, 30))


if __name__ == '__main__':
    net = DeepISP(2, 2)

    print(net.lowlevel)
    print(net.highlevel)
