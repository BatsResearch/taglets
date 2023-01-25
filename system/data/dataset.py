from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, filepaths, root, 
                transform, augmentations=None, 
                train=True, labels=None,
                label_map=None):
        """
        :param filepaths: list of images
        :param root: path to images
        :param transform: standard transform
        :param augmentations: None or tuple
        :param train: indicates in the data is in train or test folder
        :param labels: list of label
        :param label_map: dict mpping string labels to int
        """
        # Adjust filepaths
        self.train = train
        if self.train:
            self.filepaths = [f"{root}/train/{f}" for f in filepaths]
        else:
            self.filepaths = [f"{root}/test/{f}" for f in filepaths]
        
        self.transform = transform
        if augmentations:
            self.aug1_transform = augmentations[0]
            self.aug2_transform = augmentations[1]
        else:
            self.aug1_transform = None
            self.aug2_transform = None
        self.labels = labels
        self.label_map = label_map

    def __len__(self):
        # dataset size
        return len(self.filepaths)
    
    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, aug1, aug2, target) where target is index of the target class.
        """
        
        img = Image.open(self.filepaths[index]).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)            
        # Apply two transformations (strong and weak)
        if self.aug1_transform is not None:
            aug_1 = self.aug1_transform(img)
        else:
            aug_1 = img
        if self.aug2_transform is not None:
            aug_2 = self.aug2_transform(img)
        else:
            aug_2 = img
        
        # Get image label
        if self.labels is not None:
            label = int(self.label_map[self.labels[index]])
            return img, aug_1, aug_2, label, self.filepaths[index].split('/')[-1]
        else:
            return img, aug_1, aug_2, self.filepaths[index].split('/')[-1]