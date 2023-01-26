import torch
import torch.nn as nn


def Tform(I, W):
    b, c, h, w = I.shape
    # somehow (b, 3, 10) doesnt work, but (b, 10, 3) does
    W = W.reshape((b, 10, 3)).permute((0, 2, 1))

    # adding 4th channel with all 1
    I = torch.cat((I, torch.ones((b, 1, h, w))), dim=1)
    I = I.reshape((b, 1, 4, h, w))
    g = I.transpose(1, 2)

    # matmul (b, 1, 4, h, w)*(b, 4, 1, h, w) = (b, 4, 4, h, w)
    n = torch.einsum('bceij,befij->bcfij', g, I)

    # get vectorized trui
    n = torch.flatten(n, 1, 2)[:, [0, 1, 2, 3, 5, 6, 7, 10, 11, 15], :, :]
    n = n.reshape((b, 1, 10, h, w))

    # matmul (b, 3, 10)*(b, 1, 10, h, w) = (b, 3, h, w)
    n = torch.einsum('bwc,bfcij->bwij', W, n)

    return n


class DeepispLL(nn.Module):
    def __init__(self, kernel=(3, 3), stride=1, padding=1):
        super(DeepispLL, self).__init__()

        self.conv1 = nn.Conv2d(61, 61, kernel_size=kernel,
                               stride=stride, padding=padding)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(3, 3, kernel_size=kernel,
                               stride=stride, padding=padding)
        self.tanh = nn.Tanh()

    def forward(self, x):
        rh = self.relu(self.conv1(x[:, :61, :, :]))
        lh = self.tanh(self.conv2(x[:, 61:, :, :]))

        # skip connections
        lh = lh + x[:, 61:, :, :].clone()

        return torch.cat((rh, lh), 1)


class DeepispHL(nn.Module):
    def __init__(self, kernel=(3, 3), stride=2, padding=0):
        super(DeepispHL, self).__init__()

        self.conv = nn.Conv2d(64, 64, kernel_size=kernel,
                              stride=stride, padding=padding)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)

        return x


class GlobalPool2d(nn.Module):
    def __init__(self):
        super(GlobalPool2d, self).__init__()

    def forward(self, x):
        b, c, w, h = tuple(x.shape)
        return nn.AvgPool2d(kernel_size=(h, w))(x).reshape((b, c))


class DeepISP(nn.Module):
    def __init__(self, n_ll, n_hl, kernel=(3, 3), stride=1, padding=1):
        super(DeepISP, self).__init__()
        self.kernel = kernel
        self.stride = stride
        self.padding = padding

        self.lowlevel = nn.Sequential()
        self.highlevel = nn.Sequential()

        self.lowlevel.append(nn.Conv2d(3, 64, kernel_size=self.kernel,
                                       stride=self.stride,
                                       padding=self.padding))
        self.highlevel.append(nn.Conv2d(61, 64, kernel_size=self.kernel,
                                        stride=self.stride,
                                        padding=self.padding))

        for i in range(n_ll):
            self.lowlevel.append(DeepispLL(kernel=self.kernel,
                                           stride=self.stride,
                                           padding=self.padding))

        for i in range(n_hl):
            self.highlevel.append(DeepispHL(kernel=self.kernel,
                                            stride=2, padding=0))

        # append global pooling on high level to get 1x1x64 shape
        # current shape = (N/4^n_hl)*(M/4^n_hl)*64
        self.highlevel.append(GlobalPool2d())

        self.highlevel.append(nn.Linear(64, 30))

        # do some T(W, L)
        self.T = Tform

    def forward(self, x):
        I = self.lowlevel(x)
        W = self.highlevel(I[:, :61, :, :])
        x = self.T(I[:, 61:, :, :], W)
        return x


if __name__ == '__main__':
    net = DeepISP(2, 2)

    print(net.lowlevel)
    print(net.highlevel)
