import numpy as np
from os import listdir, sep
import matplotlib.pyplot as plt


def get_cp_name(epoch, checkpoint_path='/home/tima/projects/isp/deepisp/CP'):
    dirs = listdir(checkpoint_path)
    cp = [i for i in dirs if f'_e{epoch}_' in i]
    return sep.join([checkpoint_path, cp[0]])


class Norm():
    modes = {
        'simple': (-1, 1),
        'positive': (0, 1)
    }

    def __init__(self, mode='simple', bounds_after=None):
        try:
            self.a, self.b = bounds_after
        except TypeError:
            self.a, self.b = Norm.get_bounds_by_mode(mode)

    @staticmethod
    def get_bounds_by_mode(mode):
        return Norm.modes[mode]

    def __call__(self, x, bounds_before=None):
        try:
            mn, mx = bounds_before
        except TypeError:
            mn, mx = x.min(), x.max()
        
        if (self.a, self.b) == (mn, mx):
            return x

        return (self.b - self.a) * ((x - mn) / (mx - mn)) + self.a


def plot_corr(x):
    r, g, b = x[0], x[1], x[2]

    # print('x', x.shape, x.min(), x.max())

    rd = get_dist((r * 255).int().reshape(-1))
    gd = get_dist((g * 255).int().reshape(-1))
    bd = get_dist((b * 255).int().reshape(-1))

    fig, axs = plt.subplots(3, 1)

    axs[0].bar(np.arange(256), rd, color='red')
    axs[1].bar(np.arange(256), gd, color='green')
    axs[2].bar(np.arange(256), bd, color='blue')
    plt.show()


def get_dist(x):
    r = [0 for _ in range(256)]

    for i in x:
        r[i] += 1

    return np.array(r) / x.size()


if __name__ == '__main__':
    n = Norm(bounds_after=(0, 1))
    
    x = np.array([0.])
    print(n(x, bounds_before=(-1, 1)))

#     img = n(img, bounds_before=(-1, 1))
#     print(img)
