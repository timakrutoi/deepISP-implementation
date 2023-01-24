from os import listdir, sep
import numpy as np
import torch

from torchvision import transforms
# from torchvision.io import read_image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

# loading raw image
from skimage import io

# demosaicing raw image
from colour_demosaicing import demosaicing_CFA_Bayer_bilinear


class S7Dataset(Dataset):
    def __init__(self, directory, mode, target, factor, crop_size):
        self.directory = directory

        self.raw_transform = demosaicing_CFA_Bayer_bilinear
        self.crop_size = crop_size

        self.dng = '.dng'
        self.jpg = '.jpg'

        self.l = len(listdir(self.directory))

        if mode == 'train':
            self.len = 0, int(self.l * factor)
        if mode == 'test':
            self.len = int(self.l * factor), self.l

        if target == 'm':
            self.target = 'medium_exposure'
        elif target == 's':
            self.target = 'short_exposure'
            self.jpg = '1.jpg'

    def __len__(self):

        return self.len[1] - self.len[0]

    def __getitem__(self, idx):
        l = listdir(self.directory)

        i_img = io.imread(sep.join([self.directory, l[idx + self.len[0]], f'{self.target}{self.dng}'])) / 1024
        o_img = io.imread(sep.join([self.directory, l[idx + self.len[0]], f'{self.target}{self.jpg}'])) / 1024

        i_img = self.raw_transform(i_img) / 1024
        
        old_shape = i_img.shape
        new_shape = old_shape[2], self.crop_size, self.crop_size
        
        x = np.random.randint(0, old_shape[0] - self.crop_size)
        y = np.random.randint(0, old_shape[1] - self.crop_size)
        
        slice_x = slice(x, x + self.crop_size)
        slice_y = slice(y, y + self.crop_size)
        
        i_img = torch.tensor(i_img[slice_x, slice_y, :].copy())
        if (old_shape[0], old_shape[1]) != (o_img.shape[0], o_img.shape[1]):            
#             print('Bad shape detected', old_shape, o_img.shape)
            slice_x, slice_y = slice_y, slice_x
        o_img = torch.tensor(o_img[slice_x, slice_y, :].copy())
                
        i_img = i_img.reshape(new_shape)
        o_img = o_img.reshape(new_shape)
        
        # maybe do data normalization
        # img = norm(img)

        return i_img.float(), o_img.float()


def get_data(data_path, batch_size, target='m', factor=0.7, crop_size=256):
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
        shuffle=True
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
