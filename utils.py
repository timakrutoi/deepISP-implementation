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


if __name__ == '__main__':
    n = Norm(bounds_after=(0, 1))
    
    img = (np.random.randn(1, 1, 3, 3))
    img[img > 1] = 1
    img[img < -1] = -1
    # img[0, 0, 0, 0] = 1023
    # img[0, 0, 1, 1] = 512
    # img[-1, -1, -1, -1] = 0
    print(img)

    img = n(img, bounds_before=(-1, 1))

    print(img)
