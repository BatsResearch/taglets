from torch.utils.data import Dataset
from PIL import Image
import torch


class CustomDataset(Dataset):
    """
    A custom dataset used to create dataloaders.
    """
    def __init__(self, filepaths, labels=None, label_map=None, transform=None):
        """
        Create a new CustomDataset.
        
        :param root: The root directory of the images
        :param filenames: A list of filenames
        :param labels: A list of labels
        :param transform: A transform to perform on the images
        """
        self.filepaths = filepaths
        self.labels = labels
        self.label_map = label_map
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.filepaths[index]).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.labels:
            if self.label_map is not None:
                label = torch.tensor(self.label_map[(self.labels[index])])
            else:
                label = torch.tensor(int(self.labels[index]))
            return img, label
        else:
            return img

    def __len__(self):
        return len(self.filepaths)


class SoftLabelDataset(Dataset):
    """
    A custom dataset used to create dataloaders.
    """
    def __init__(self, dataset, labels, remove_old_labels=False):
        """
        Create a new SoftLabelDataset.
        :param images: A list of filenames
        :param labels: A list of labels
        """
        self.dataset = dataset
        self.labels = labels
        self.remove_old_labels = remove_old_labels

    def __getitem__(self, index):
        data = self.dataset[index]
        label = self.labels[index]
        
        if self.remove_old_labels:
            data = data[0]
            
        return data, label

    def __len__(self):
        return len(self.dataset)
