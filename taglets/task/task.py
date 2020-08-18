import copy
import torch
import torchvision.models as models


class Task:
    """
    A class defining an image classification task
    """
    def __init__(self, name, classes, input_shape, labeled_train_data, unlabeled_train_data, validation_data,
                 whitelist=None, scads_path=None, scads_embedding_path=None, unlabeled_test_data=None):
        """
        Create a new Task

        :param name: a human-readable name for this task
        :param classes: list of SCADS node IDs
        :param labeled_train_data: DataLoader for labeled training data
        :param unlabeled_train_data: DataLoader for unlabeled training data
        :param validation_data: DataLoader for labeled validation data
        """
        self.name = name
        self.description = ''
        self.classes = classes
        self.input_shape = input_shape
        self.labeled_train_data = labeled_train_data
        self.unlabeled_train_data = unlabeled_train_data
        self.validation_data = validation_data
        self.scads_path = scads_path
        self.scads_embedding_path = scads_embedding_path
        self.unlabeled_test_data = unlabeled_test_data

        self.initial = models.resnet18(pretrained=True)
        self.initial.fc = torch.nn.Identity()
        self.whitelist = whitelist

    def get_labeled_train_data(self):
        return self.labeled_train_data

    def get_unlabeled_data(self, train=True):
        if self.unlabeled_test_data is not None and not train:
            return self.unlabeled_test_data
        else:
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
