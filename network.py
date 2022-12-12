#!/usr/bin/python3
#encode=utf-8

import torch
import torch.nn as nn

import numpy as np


class DeepispLL(nn.Module):
    def __init__(self, kernel=(3,3), stride=1, padding=1):
        super(DeepispLL, self).__init__()

        # self.size = n * m
        self.padding = padding
        self.kernel = kernel
        self.stride = stride

    def forward(self, x):
        rh = nn.Conv2d(61, 61, kernel_size=self.kernel, stride=self.stride, padding=self.padding)(x[:,:61,:,:])
        rh = nn.ReLU()(rh)

        lh = nn.Conv2d(3, 3, kernel_size=self.kernel, stride=self.stride, padding=self.padding)(x[:,61:,:,:])
        lh = nn.Tanh()(lh)

        # need to so some sum
        # lh += x

        return torch.cat((rh, lh), 1)


class DeepispHL(nn.Module):
    def __init__(self, kernel=(3,3), stride=2, padding=1):
        super(DeepispHL, self).__init__()

        self.padding = padding
        self.kernel = kernel
        self.stride = stride

    def forward(self, x):
        x = nn.Conv2d(64, 64, kernel_size=self.kernel, stride=self.stride, padding=self.padding)(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(2, 2)(x)

        return x


class GlobalPool2d(nn.Module):
    def __init__(self):
        super(GlobalPool2d, self).__init__()

    def forward(self, x):
        b, c, w, h = tuple(x.shape)
        return nn.AvgPool2d(kernel_size=(h,w))(x).reshape((b, c))


def triu(rgb):
    res = torch.tensor(np.empty(10), dtype=torch.float)
    r, g, b = rgb[0, 0], rgb[0, 1], rgb[0, 2]
    res[0] = r*r
    res[1] = r*g
    res[2] = r*b
    res[3] = r
    res[4] = g*g
    res[5] = g*b
    res[6] = g
    res[7] = b*b
    res[8] = b
    res[9] = 1

    return res.reshape((10))


def Tform(I, W):
    b, c, h, w = I.shape
    res = torch.tensor(np.empty(I.shape))
    W = W.reshape((3, 10))
    for x in range(h):
        for y in range(w):
            res[:, :, x, y] = W @ triu(I[:, :, x, y])
    return res


class DeepISP(nn.Module):
    def __init__(self, n_ll, n_hl, stride=1, padding=1):
        super(DeepISP, self).__init__()
        self.stride = stride
        self.padding = padding
        
        self.lowlevel = nn.Sequential()
        self.highlevel = nn.Sequential()

        self.lowlevel.append(nn.Conv2d(3, 64, kernel_size=(3,3), stride=self.stride, padding=self.padding))
        self.highlevel.append(nn.Conv2d(61, 64, kernel_size=(3,3), stride=self.stride, padding=self.padding))

        for i in range(n_ll):
            self.lowlevel.append(DeepispLL(stride=self.stride, padding=self.padding))

        for i in range(n_hl):
            self.highlevel.append(DeepispHL(stride=self.stride, padding=self.padding))

        # append global pooling on high level to get 1x1x64 shape
        # current shape = (N/4^n_hl)*(M/4^n_hl)*64
        # self.highlevel.append(nn.MaxPool2(...))
        self.highlevel.append(GlobalPool2d())

        self.highlevel.append(nn.Linear(64, 30))
        
        # do some T(W, L)
        self.T = Tform
    
    def forward(self, x):
        I = self.lowlevel(x)
        W = self.highlevel(I[:,:61,:,:])
        return self.T(I[:,61:,:,:], W)


if __name__ == '__main__':
    net = DeepISP(2, 2)

    print(net.lowlevel)
    print(net.highlevel)
