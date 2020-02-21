import copy
import os
import torch
import torchvision.transforms as transforms
from taglets.data.custom_dataset import CustomDataSet
from torch.utils import data


class Task:
    """
    A class defining an image classification task
    """
    def __init__(self, name, classes, labeled_train_data, unlabeled_train_data,
                 validation_data):
        """
        Create a new Task

        :param name: a human-readable name for this task
        :param classes: map from DataLoader class labels to SCADS node IDs
        :param labeled_train_data: DataLoader for labeled training data
        :param unlabeled_train_data: DataLoader for unlabeled training data
        :param validation_data: DataLoader for labeled validation data
        """
        self.name = name
        self.description = ''
        self.classes = classes
        self.labeled_train_data = labeled_train_data
        self.unlabeled_train_data = unlabeled_train_data
        self.validation_data = validation_data

        self.initial = None

    def get_classes(self):
        """
        :return: a copy of the map from DataLoader class labels to SCADS node IDs
        """
        return dict(self.classes)

    def get_labeled_train_data(self):
        return self.labeled_train_data

    def get_unlabeled_train_data(self):
        return self.unlabeled_train_data

    def get_validation_data(self):
        return self.validation_data

    def set_initial_model(self, initial):
        """
        Sets an initial model on the task that will be used as the architecture
        and initial weights of models created for the task, including taglets and
        the end model

        :param initial: the initial model
        """
        self.initial = initial

    def get_initial_model(self):
        """
        Returns a deep copy of the task's initial model

        :return: copy of the initial model, or None if no model is set
        """
        if self.initial is None:
            return None
        else:
            return copy.deepcopy(self.initial)

    def get_labeled_images_list(self):
        """get list of image names and labels"""
        image_names = [img_name for img_name, label in self.labeled_images]
        image_labels = [label for img_name, label in self.labeled_images]
        return image_names, image_labels

    def get_unlabeled_image_names(self):
        """return list of name of unlabeled images"""
        labeled_image_names = {img_name for img_name, label in self.labeled_images}
        unlabeled_images_names = []
        for img in os.listdir(self.unlabeled_image_path):
            if img not in labeled_image_names:
                unlabeled_images_names.append(img)
        return unlabeled_images_names
    
    def get_test_image_names(self):
        """return list of name of test images"""
        test_images_names = []
        for img in os.listdir(self.evaluation_image_path):
            test_images_names.append(img)
        return test_images_names
