import numpy as np
from scipy.ndimage import convolve
import torch


def masks_bayer(shape, pattern):
    channels = {channel: np.zeros(shape, dtype="bool") for channel in "RGB"}
    for channel, (y, x) in zip(pattern, [(0, 0), (0, 1), (1, 0), (1, 1)]):
        channels[channel][y::2, x::2] = 1

    return tuple(channels.values())


def demosaicing_Bayer_bilinear(CFA, pattern="RGGB"):
    R_m, G_m, B_m = masks_bayer(CFA.shape, pattern)

    H_G = torch.tensor([
            [0, 1, 0],
            [1, 4, 1],
            [0, 1, 0],
        ], dtype=torch.float32)

    H_RB = torch.tensor([
            [1, 2, 1],
            [2, 4, 2],
            [1, 2, 1],
        ], dtype=torch.float32)

    R = convolve(CFA * R_m, H_RB / 4, mode='mirror')
    G = convolve(CFA * G_m, H_G / 4, mode='mirror')
    B = convolve(CFA * B_m, H_RB / 4, mode='mirror')

    R = torch.tensor(R)
    G = torch.tensor(G)
    B = torch.tensor(B)

    del R_m, G_m, B_m, H_RB, H_G

    return torch.stack([R, G, B], 2)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    s = 10
    img = np.ones((s, s))
    pattern = 'RGGB'

    deb = demosaicing_Bayer_bilinear(img, pattern)

    plt.imshow(deb)
    plt.show()
