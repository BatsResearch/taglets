import numpy as np
from pathlib import Path
from PIL import Image
import os
import torch
import torchvision.transforms as transforms
from custom_dataset import CustomDataSet
from torch.utils import data

from modules.module import BaseModule
from taglet_executer import TagletExecuter


class Task:
    """ Task class """

    def __init__(self, metadata):
        self.description = ''
        self.problem_type = metadata['problem_type']
        self.task_id = metadata['task_id']
        self.classes = []
        self.evaluation_image_path = "path to test images"
        self.unlabeled_image_path = 'path to unlabeled images'
        self.labeled_images = []    # A list of tuples with name and label e.g., ['1.png', '2'], ['2.png', '7'], etc.
        self.number_of_channels = None
        self.train_data_loader = None


    def add_labeled_images(self, new_labeled_images):
        self.labeled_images.extend(new_labeled_images)

    def _transform_image(self):
        if self.number_of_channels == 3:
            data_mean = [0.485, 0.456, 0.406]
            data_std = [0.229, 0.224, 0.225]
        elif self.number_of_channels == 1:
            data_mean = [0.5]
            data_std = [0.5]

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=data_mean, std=data_std)
        ])
        return transform

    def load_labeled_data(self, batch_size, num_workers):

        transform = self._transform_image()
        image_names = [img_name for img_name, label in self.labeled_images]
        image_labels = [label for img_name, label in self.labeled_images]
        train_val_test_data = CustomDataSet(self.unlabeled_image_path,
                                            image_names,
                                            image_labels,
                                            transform,
                                            self.number_of_channels)

        # 70% for training, 15% for validation, and 15% for test
        train_percent = 0.7
        num_data = len(train_val_test_data)
        indices = list(range(num_data))
        train_split = int(np.floor(train_percent * num_data))
        np.random.shuffle(indices)
        train_idx = indices[:train_split]
        val_test_data = indices[train_split:]
        val_split = int(np.floor(len(val_test_data) / 2))
        valid_idx = val_test_data[:val_split]
        test_idx = val_test_data[val_split:]

        train_set = data.Subset(train_val_test_data, train_idx)
        val_set = data.Subset(train_val_test_data, valid_idx)
        test_set = data.Subset(train_val_test_data, test_idx)

        # test_data = datasets.ImageFolder(self.task.test_image_path, transform= transform)

        train_data_loader = torch.utils.data.DataLoader(train_set,
                                                        batch_size=batch_size,
                                                        shuffle=False,
                                                        num_workers=num_workers)
        val_data_loader = torch.utils.data.DataLoader(val_set,
                                                      batch_size=batch_size,
                                                      shuffle=False,
                                                      num_workers=num_workers)
        test_data_loader = torch.utils.data.DataLoader(test_set,
                                                       batch_size=batch_size,
                                                       shuffle=False,
                                                       num_workers=num_workers)

        print('number of training data: %d' % len(train_data_loader.dataset))
        print('number of validation data: %d' % len(val_data_loader.dataset))
        print('number of test data: %d' % len(test_data_loader.dataset))

        return train_data_loader,val_data_loader,test_data_loader

    def get_unlabeled_images(self ,batch_size, num_workers):

        transform = self._transform_image()
        labeled_image_names = [img_name for img_name, label in self.labeled_images]
        unlabeled_images_names = []

        for img in os.listdir(self.unlabeled_image_path):
            if img not in labeled_image_names:
                unlabeled_images_names.append(img)


        unlabeled_data = CustomDataSet(self.unlabeled_image_path,
                                            unlabeled_images_names,
                                            None,
                                            transform,
                                            self.number_of_channels)


        unlabeled_data_loader = torch.utils.data.DataLoader(unlabeled_data,
                                                        batch_size=batch_size,
                                                        shuffle=False,
                                                        num_workers=num_workers)

        return unlabeled_data_loader

