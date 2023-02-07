import torch
import torch.nn as nn


def Tform(I, W):
    b, c, h, w = I.shape
    # reshape W from (b, 30) to (b, 3, 10)
    # somehow (b, 3, 10) doesnt work, but (b, 10, 3) does
    # W = W.reshape(b, 3, 10)
    W = W.reshape((b, 10, 3)).transpose(1, 2)
    W += torch.tensor([[
        # rr rg rb  r gg gb  g bb  b  1
        # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # r
        # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # g
        # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # b
        # [0, 0, 0, 0.6077, 0, 0, -0.0125, 0,  0.5503,  0.2050],
        # [0, 0, 0, 0.6448, 0, 0,  0.4843, 0,  0.2517,  0.3991],
        # [0, 0, 0, 0.1757, 0, 0,  0.1883, 0,  0.9128, -0.0028]
        # [-0.2667,  0.1698, -0.1399,  0.2564, -0.3247, -0.1434,  0.2817, -0.3895, 0.1804,  0.2732],
        # [-0.2325,  0.0820, -0.2358,  0.3804,  0.0957,  0.1536,  0.1814, -0.6373, 0.3890,  0.6108],
        # [-0.1638,  0.1027, -0.4943,  0.5214,  0.0169, -0.1929,  0.0512,  0.0107, 0.2826,  0.1593]
        # [0, 0, 0, 0.0848, 0, 0, 0.0508, 0, 0.0846,  0.0117],
        # [0, 0, 0, 0.1285, 0, 0, 0.0977, 0, 0.1284, -0.0482],
        # [0, 0, 0, 0.1629, 0, 0, 0.1408, 0, 0.1628, -0.1052]
        [-0.0463, -0.0476, -0.0464,  0.0504, -0.0350, -0.0475,  0.0181, -0.0463, 0.0503,  0.0396],
        [-0.0710, -0.0736, -0.0711,  0.0828, -0.0632, -0.0736,  0.0531, -0.0710, 0.0827, -0.0060],
        [-0.0890, -0.0949, -0.0890,  0.1096, -0.0899, -0.0949,  0.0872, -0.0890, 0.1095, -0.0520]
    ]], dtype=torch.float)

    # adding 4th channel with all 1
    I = torch.cat((I, torch.ones((b, 1, h, w), dtype=torch.float)), dim=1)

    # matmul (b, 4, h, w)*(b, 4, h, w) = (b, 4, 4, h, w)
    n = torch.einsum('beij,bfij->bfeij', I, I)

    # get vectorized triu
    triu = [0, 1, 2, 3, 5, 6, 7, 10, 11, 15]
    n = torch.flatten(n, 1, 2)[:, triu]

    # matmul (b, 3, 10)*(b, 10, h, w) = (b, 3, h, w)
    n = torch.einsum('bwc,bcij->bwij', W, n)

    # wtf?
    if torch.any(n > 1) or torch.any(n < -1):
        # print(f'Bad values {n.min()} - {n.max()}')
        n[n > 1] = 1
        n[n < -1] = -1

    return n


class DeepispLL(nn.Module):
    def __init__(self, in_channels=64, kernel=(3, 3), stride=1, padding=1):
        super(DeepispLL, self).__init__()

        self.in_channels = in_channels
        self.img_channels = 3

        self.conv1 = nn.Conv2d(self.in_channels,
                               64 - self.img_channels,
                               kernel_size=kernel,
                               stride=stride, padding=padding)

        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(self.in_channels, self.img_channels,
                               kernel_size=kernel,
                               stride=stride, padding=padding)
        self.tanh = nn.Tanh()

    def forward(self, x):
        rh = self.relu(self.conv1(x))
        lh = self.tanh(self.conv2(x))

        # skip connections
        lh = lh + x[:, -self.img_channels:].clone()

        x = torch.cat((rh, lh), 1)

        return x


class DeepispHL(nn.Module):
    def __init__(self, in_channels=64, kernel=(3, 3), stride=2, padding=1):
        super(DeepispHL, self).__init__()

        self.conv = nn.Conv2d(in_channels, 64, kernel_size=kernel,
                              stride=stride, padding=padding)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, x):
        x = self.pool(self.relu(self.conv(x)))

        return x


# [TODO] fix "Given input size: (64x1x1).
# Calculated output size: (64x0x0). Output size is too small"
# when nhl is too big for img
class GlobalPool2d(nn.Module):
    def __init__(self):
        super(GlobalPool2d, self).__init__()

    def forward(self, x):
        b, c, w, h = x.shape
        return nn.functional.adaptive_avg_pool2d(x, 1).reshape((b, c))


class DeepISP(nn.Module):
    def __init__(self, n_ll, n_hl):
        super(DeepISP, self).__init__()

        assert n_ll >= 1, f'Nll must be greater than 0 (current is {n_ll})'
        assert n_hl >= 1, f'Nhl must be greater than 0 (current is {n_hl})'

        self.lowlevel = nn.Sequential()
        self.lowlevel.append(DeepispLL(in_channels=3))
        for i in range(n_ll - 1):
            self.lowlevel.append(DeepispLL())

        self.highlevel = nn.Sequential()
        self.highlevel.append(DeepispHL(in_channels=61))
        for i in range(n_hl - 1):
            self.highlevel.append(DeepispHL())

        # append global pooling on high level to get 1x1x64 shape
        # current shape is (N/4^n_hl, M/4^n_hl, 64)
        # self.highlevel.append(GlobalPool2d())
        self.highlevel.append(GlobalPool2d())

        self.highlevel.append(nn.Linear(64, 30))
        # with torch.no_grad():
        #     print(self.highlevel[-1].weight.shape)
        #     self.highlevel[-1].weight.copy_(torch.zeros((30, 64)))
        #     self.highlevel[-1].bias.copy_(torch.zeros(3, 10).view(-1))

        self.T = Tform

    def forward(self, x):
        I = self.lowlevel(x)
        W = self.highlevel(I[:, :-3])
        x = self.T(I[:, -3:], W)
        return x


if __name__ == '__main__':
    net = DeepISP(2, 2)

    print(net.lowlevel)
    print(net.highlevel)
