from os import listdir, sep
import torch

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

# loading raw image
from skimage import io

# demosaicing raw image
from demosaic import demosaicing_Bayer_bilinear

from utils import Norm


def random_crop(initial, target, crop_size):
    x = torch.randint(low=0, high=initial.shape[0] - crop_size, size=(1,))
    y = torch.randint(low=0, high=initial.shape[1] - crop_size, size=(1,))
    # print(x, y)

    # for blue color
    # x, y = 1926, 727
    # x, y = 1138, 1172

    slice_x = slice(x, x + crop_size)
    slice_y = slice(y, y + crop_size)

    # this *if* checks if jpg version of image is rotated
    # on 90 degrees (raw data in never rotated)
    if (initial.shape[:2]) != (target.shape[:2]):
        # print('Bad shape detected (', idx, ')', i_img.shape, o_img.shape)
        target = target.rot90(k=1, dims=[0, 1])

    initial = initial[slice_x, slice_y, :]
    target = target[slice_x, slice_y, :]

    return initial, target


def random_flip(initial, target, flip_mode):
    f = []

    if 'h' in flip_mode and torch.randint(low=0, high=1, size=(1,)):
        f.append(1)

    if 'v' in flip_mode and torch.randint(low=0, high=1, size=(1,)):
        f.append(0)

    initial = initial.flip(dims=f)
    target = target.flip(dims=f)

    return initial, target


class S7Dataset(Dataset):
    def __init__(self, directory, mode, target,
                 factor, crop_size, norm_mode, flip):
        self.directory = directory

        self.raw_transform = demosaicing_Bayer_bilinear
        self.crop_size = crop_size
        self.flip = flip

        self.dng = '.dng'
        self.jpg = '.jpg'

        tmp_len = len(listdir(self.directory))

        if mode == 'train':
            self.start = 0
            self.len = int(tmp_len * factor)
        elif mode == 'test':
            self.start = int(tmp_len * factor)
            self.len = tmp_len - self.start

        if target == 'm':
            self.target = 'medium_exposure'
        elif target == 's':
            self.target = 'short_exposure'
            self.dng = '1.dng'

        self.norm = Norm(mode=norm_mode)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        dirs = listdir(self.directory)

        i1p = sep.join([self.directory,
                        dirs[idx + self.start],
                        f'{self.target}{self.dng}'])
        i2p = sep.join([self.directory,
                        dirs[idx + self.start],
                        f'{self.target}{self.jpg}'])

        # values from 0 to 1023
        i_img = torch.tensor(io.imread(i1p).astype('float'))
        i_img = self.norm(i_img, bounds_before=(0, 1023))

        # values from 0 to 255
        o_img = torch.tensor(io.imread(i2p).astype('float'))
        o_img = self.norm(o_img, bounds_before=(0, 255))

        patterns = ["RGGB", "BGGR", "GRBG", "GBRG"]
        i_img = self.raw_transform(i_img, pattern=patterns[0])

        if self.crop_size is not None:
            i_img, o_img = random_crop(i_img, o_img, self.crop_size)

        if self.flip is not None:
            i_img, o_img = random_flip(i_img, o_img, self.flip)

        i_img = i_img.permute(2, 0, 1)
        o_img = o_img.permute(2, 0, 1)
        # i_img = i_img[[2,1,0]]
        # o_img = o_img[[2,1,0]]

        return i_img, o_img


def get_data(data_path, batch_size,
             target='m', factor=0.7,
             crop_size=256, flip='hv',
             norm_mode='simple',
             num_workers=0):

    # [TODO] get rid of those long args
    train_data = S7Dataset(
        directory=data_path,
        mode='train',
        target=target,
        factor=factor,
        crop_size=crop_size,
        flip=flip,
        norm_mode=norm_mode
    )

    test_data = S7Dataset(
        directory=data_path,
        mode='test',
        target=target,
        factor=factor,
        crop_size=crop_size,
        flip=flip,
        norm_mode=norm_mode
    )

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, test_loader


if __name__ == '__main__':
    path = '../dataset/S7-ISP-Dataset'
    train, test = get_data(path)

    for i in train:
        print(i)
