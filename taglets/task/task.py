import copy
import torch
import torchvision.models as models


class Task:
    """
    A class defining an image classification task
    """
    def __init__(self, name, classes, input_shape, labeled_train_data, unlabeled_train_data,
                 validation_data, whitelist,scads_path):
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
        self.input_shape = input_shape
        self.labeled_train_data = labeled_train_data
        self.unlabeled_train_data = unlabeled_train_data
        self.validation_data = validation_data
        self.scads_path = scads_path

        self.initial = models.resnet18(pretrained=True)
        self.initial.fc = torch.nn.Identity()
        self.whitelist = whitelist
        self.train_label_distr = Task._build_label_distr(self.labeled_train_data)
        self.validation_label_distr = Task._build_label_distr(self.validation_data)

    @staticmethod
    def _build_label_distr(dataset):
        label_distr = {}
        # possibly naive?
        for i in range(len(dataset)):
            _, label = dataset[i]
            if label not in label_distr:
                label_distr[label] = 0
            label_distr[label] += 1
        return label_distr

    def get_classes(self):
        """
        :return: a copy of the map from DataLoader class labels to SCADS node IDs
        """
        return dict(self.classes)

    def get_labeled_train_data(self):
        return self.labeled_train_data

    def get_train_label_distr(self):
        return self.train_label_distr

    def get_train_labels(self):
        return [int(x) for x in self.labeled_train_data.labels]

    def get_unlabeled_train_data(self):
        return self.unlabeled_train_data

    def get_validation_data(self):
        return self.validation_data

    def get_validation_label_distr(self):
        return self.validation_label_distr

    def get_validation_labels(self):
        return [int(x) for x in self.validation_data.labels]

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
