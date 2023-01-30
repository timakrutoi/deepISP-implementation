from os import listdir, sep
import numpy as np
import torch

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

# loading raw image
from skimage import io

# demosaicing raw image
from demosaic import demosaicing_Bayer_bilinear

from kornia.color import rgb_to_ycbcr


class S7Dataset(Dataset):
    def __init__(self, directory, mode, target,
                 factor, crop_size):
        self.directory = directory

        self.raw_transform = demosaicing_Bayer_bilinear
        self.crop_size = crop_size

        self.dng = '.dng'
        self.jpg = '.jpg'

        tmp_len = len(listdir(self.directory))

        if mode == 'train':
            self.start = 0
            self.len = int(tmp_len * factor)
        if mode == 'test':
            self.start = int(tmp_len * factor)
            self.len = tmp_len - self.start

        if target == 'm':
            self.target = 'medium_exposure'
        elif target == 's':
            self.target = 'short_exposure'
            self.dng = '1.dng'

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
        i_img = (i_img - 512) / 512

        # values from 0 to 255
        o_img = torch.tensor(io.imread(i2p).astype('float'))
        o_img = (o_img - 128) / 128

        i_img = self.raw_transform(i_img)

        x = np.random.randint(0, i_img.shape[0] - self.crop_size)
        y = np.random.randint(0, i_img.shape[1] - self.crop_size)
        # print(x, y)

        # for blue color
        # x, y = 1926, 727

        slice_x = slice(x, x + self.crop_size)
        slice_y = slice(y, y + self.crop_size)

        # this *if* checks if jpg version of image is rotated
        # on 90 degrees (raw data in never rotated)
        if (i_img.shape[:2]) != (o_img.shape[:2]):
            # print('Bad shape detected (', idx, ')', i_img.shape, o_img.shape)
            o_img = o_img.rot90(k=1, dims=[0, 1])

        i_img = i_img[slice_x, slice_y, :]
        o_img = o_img[slice_x, slice_y, :]

        i_img = i_img.permute(2, 0, 1)
        o_img = o_img.permute(2, 0, 1)

        return i_img, o_img


def get_data(data_path, batch_size,
             target='m', factor=0.7,
             crop_size=256, norm=False,
             num_workers=0):

    train_data = S7Dataset(
        directory=data_path,
        mode='train',
        target=target,
        factor=factor,
        crop_size=crop_size
    )

    test_data = S7Dataset(
        directory=data_path,
        mode='test',
        target=target,
        factor=factor,
        crop_size=crop_size
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
