from scipy.ndimage import convolve
import numpy as np


def masks_bayer(shape, pattern):
    
    channels = {channel: np.zeros(shape, dtype="bool") for channel in "RGB"}
    for channel, (y, x) in zip(pattern, [(0, 0), (0, 1), (1, 0), (1, 1)]):
        channels[channel][y::2, x::2] = 1

    return tuple(channels.values())


def demosaicing_Bayer_bilinear(x, pattern="RGGB"):

    CFA = x.astype('float')
    R_m, G_m, B_m = masks_bayer(CFA.shape, pattern)

    H_G = np.array([
            [0, 1, 0],
            [1, 4, 1],
            [0, 1, 0],
        ], dtype='float')

    H_RB = np.array([
            [1, 2, 1],
            [2, 4, 2],
            [1, 2, 1],
        ], dtype='float')

    R = convolve(CFA * R_m, H_RB / 4, mode='mirror')
    G = convolve(CFA * G_m, H_G  / 4, mode='mirror')
    B = convolve(CFA * B_m, H_RB / 4, mode='mirror')

    del R_m, G_m, B_m, H_RB, H_G

    return np.stack([R, G, B], 2)


if __name__ == '__main__':
    img_shape = (3,3)
    # img = np.ones(img_shape)
    pattern = 'RGGB'
    
    masks = masks_bayer(img_shape, pattern)

    for i in masks:
        print(i)

    s = 10
    f = 1023.
    img = np.ones((s, s))
    deb = my_bil(img).reshape((s, s, 3))

    plt.imshow(deb)
    plt.show()
