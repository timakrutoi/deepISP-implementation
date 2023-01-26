from os import listdir, sep
import numpy as np
import torch

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

# loading raw image
from skimage import io

# demosaicing raw image
from demosaic import demosaicing_Bayer_bilinear


class S7Dataset(Dataset):
    def __init__(self, directory, mode, target,
                 factor, crop_size, norm, rand=False):
        self.directory = directory

        self.raw_transform = demosaicing_Bayer_bilinear
        self.crop_size = crop_size
        self.norm = norm
        self.rand = rand

        self.dng = '.dng'
        self.jpg = '.jpg'

        tmp_len = len(listdir(self.directory))

        if mode == 'train':
            self.len = 0, int(tmp_len * factor)
        if mode == 'test':
            self.len = int(tmp_len * factor), tmp_len

        if target == 'm':
            self.target = 'medium_exposure'
        elif target == 's':
            self.target = 'short_exposure'
            self.jpg = '1.jpg'

    def __len__(self):
        return self.len[1] - self.len[0]

    def __getitem__(self, idx):
        dirs = listdir(self.directory)

        i1p = sep.join([self.directory,
                        dirs[idx + self.len[0]],
                        f'{self.target}{self.dng}'])
        i2p = sep.join([self.directory,
                        dirs[idx + self.len[0]],
                        f'{self.target}{self.jpg}'])

        i_img = torch.tensor(io.imread(i1p).astype('float'))
        o_img = torch.tensor(io.imread(i2p).astype('float'))

        if self.norm:
            i_img = (i_img - 512) / 512
            o_img = (o_img - 128) / 128

        i_img = self.raw_transform(i_img)

        old_shape = i_img.shape

        if self.rand:
            x = np.random.randint(0, old_shape[0] - self.crop_size)
            y = np.random.randint(0, old_shape[1] - self.crop_size)
        else:
            # debugging thing
            x, y = 1779, 982

        slice_x = slice(x, x + self.crop_size)
        slice_y = slice(y, y + self.crop_size)

        i_img = i_img[slice_x, slice_y, :].clone().detach()
        if (old_shape[0], old_shape[1]) != (o_img.shape[0], o_img.shape[1]):
            # print('Bad shape detected', old_shape, o_img.shape)
            slice_x, slice_y = slice_y, slice_x
        o_img = o_img[slice_x, slice_y, :].clone().detach()

        i_img = i_img.permute(2, 0, 1)
        o_img = o_img.permute(2, 0, 1)

        return i_img.float(), o_img.float()


def get_data(data_path, batch_size,
             target='m', factor=0.7,
             crop_size=256, norm=False):

    train_data = S7Dataset(
        directory=data_path,
        mode='train',
        target=target,
        factor=factor,
        crop_size=crop_size,
        norm=norm
    )

    test_data = S7Dataset(
        directory=data_path,
        mode='test',
        target=target,
        factor=factor,
        crop_size=crop_size,
        norm=norm
    )

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=False
    )
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, test_loader


if __name__ == '__main__':
    path = '../dataset/S7-ISP-Dataset'
    train, test = get_data(path)

    for i in train:
        print(i)
