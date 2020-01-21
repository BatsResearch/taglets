from torch.utils.data import Dataset
from PIL import Image
import os
import torch


class CustomDataSet(Dataset):
    def __init__(self, root, images, labels, transform, num_channels):
        self.root = root
        self.images = images
        self.labels = labels
        self.transform = transform
        self.num_channels = num_channels

    def __getitem__(self, index):
        img = os.path.join(self.root, self.images[index])
        img = Image.open(img)
        if self.num_channels == 3:
            img = img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        label = torch.tensor(int(self.labels[index]))
        # Return img, self.labels[index], index
        return img, label, index

    def __len__(self):
        return len(self.images)
