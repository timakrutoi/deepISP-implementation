from torch.utils import Dataset
from torchvision.io import read_image
from torchvision import transforms
from torch.utils.data import DataLoader


class S7Dataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])

        image = transforms.Resize((300, 300))(read_image(img_path))

        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image.float(), label


def get_data(data_path, batch_size=64):
    train_data = S7Dataset(
        annotations_file=data_path + 'annotations_train.csv',
        img_dir=data_path + 'images'
    )

    test_data = S7Dataset(
        annotations_file=data_path + 'annotations_test.csv',
        img_dir=data_path + 'images'
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
