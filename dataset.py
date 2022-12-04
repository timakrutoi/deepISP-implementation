from os import listdir, sep

from torchvision import transforms
# from torchvision.io import read_image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

# loading raw image
from skimage import io

# demosaicing raw image
from colour_demosaicing import demosaicing_CFA_Bayer_bilinear


class S7Dataset(Dataset):
    def __init__(self, directory, mode, target='m', factor=0.7):
        self.directory = directory

        self.raw_transform = demosaicing_CFA_Bayer_bilinear

        self.dng = '.dng'
        self.jpg = '.jpg'

        self.l = len(listdir(self.directory))

        if mode == 'train':
            self.len = 0, int(self.l * factor)
        if mode == 'test':
            self.len = self.l - int(self.l * factor), self.l

        if target == 'm':
            self.target = 'medium_exposure'
        elif target == 's':
            self.target = 'short_exposure'
            self.jpg = '1.jpg'


    def __len__(self):
        return self.len[1] - self.len[0]

    def __getitem__(self, idx):
        l = listdir(self.directory)

        # i_img = read_image(sep.join([self.directory, l[idx + self.len[0]], f'{self.target}{self.dng}']))
        # o_img = read_image(sep.join([self.directory, l[idx + self.len[0]], f'{self.target}{self.jpg}']))

        i_img = io.imread(sep.join([self.directory, l[idx + self.len[0]], f'{self.target}{self.dng}']))
        o_img = io.imread(sep.join([self.directory, l[idx + self.len[0]], f'{self.target}{self.jpg}']))

        i_img = self.raw_transform(i_img)

        return i_img.astype('float64'), o_img.astype('float64')


def get_data(data_path='../dataset/S7-ISP-Dataset', batch_size=64):
    train_data = S7Dataset(
        directory=data_path,
        mode='train'
    )

    test_data = S7Dataset(
        directory=data_path,
        mode='test'
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
