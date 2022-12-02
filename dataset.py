from os import listdir, sep

from torchvision import transforms
from torchvision.io import read_image
from torch.utils import Dataset
from torch.utils.data import DataLoader


class S7Dataset(Dataset):
    def __init__(self, directory, mode='m', transform=None, target_transform=None):
        self.directory = directory

        self.transform = transform
        self.target_transform = target_transform

        self.dng = '.dng'
        self.jpg = '.jpg'

        if mode == 'm':
            self.mode = 'medium_exposure'
        elif mode == 's':
            self.mode = 'short_exposure'
            self.jpg = '1.jpg'


    def __len__(self):
        return len(os.listdir(self.directory))

    def __getitem__(self, idx):
        l = os.listdir(self.directory)
        i_img = read_image(sep.join([self.directory, l[idx]], f'{self.mode}{self.dng}'))
        o_img = read_image(sep.join([self.directory, l[idx]], f'{self.mode}{self.jpg}'))

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return i_img.float(), o_img.float()


def get_data(data_path='../dataset/S7-ISP-Dataset', batch_size=64):
    train_data = S7Dataset(
        img_dir=data_path
    )

    test_data = S7Dataset(
        img_dir=data_path
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
