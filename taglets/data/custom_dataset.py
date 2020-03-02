from torch.utils.data import Dataset
from PIL import Image
import os
import torch


class CustomDataSet(Dataset):
    """
    A custom dataset used to create dataloaders.
    """
    def __init__(self, root, filenames, labels, transform, num_channels):
        """
        Create a new CustomDataSet.
        
        :param root: The root directory of the images
        :param filenames: A list of filenames
        :param labels: A list of labels
        :param transform: A transform to perform on the images
        :param num_channels: The number of channels
        """
        self.root = root
        self.filenames = filenames
        self.labels = labels
        self.transform = transform
        self.num_channels = num_channels

    def __getitem__(self, index):
        img = os.path.join(self.root, self.filenames[index])
        img = Image.open(img)
        if self.num_channels == 3:
            img = img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.labels:
            label = torch.tensor(int(self.labels[index]))
            return img, label
        else:
            return img

    def __len__(self):
        return len(self.filenames)


class SoftLabelDataSet(Dataset):
    """
    A custom dataset used to create dataloaders.
    """
    def __init__(self, images, labels):
        """
        Create a new CustomDataSet.
        :param images: A list of filenames
        :param labels: A list of labels
        """
        self.images = images
        self.labels = labels

    def __getitem__(self, index):
        img = self.images[index]
        label = self.labels[index]

        return img, label

    def __len__(self):
        return len(self.images)

