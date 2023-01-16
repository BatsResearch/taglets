import copy
import torch
import torchvision.models as models


class Task:
    """
    A class defining an image classification task
    """
    def __init__(self, name, classes, input_shape, labeled_train_data, unlabeled_train_data, validation_data, 
                batch_size=128, whitelist=None, scads_path=None, scads_embedding_path=None, 
                processed_scads_embedding_path=None, unlabeled_test_data=None, unlabeled_train_labels=None, 
                video_classification=False, wanted_num_related_class=None):
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
        self.batch_size = batch_size
        self.scads_path = scads_path
        self.scads_embedding_path = scads_embedding_path
        self.processed_scads_embedding_path = processed_scads_embedding_path
        self.unlabeled_test_data = unlabeled_test_data
        self.unlabeled_train_labels = unlabeled_train_labels
        self.video_classification = video_classification
        self.wanted_num_related_class = wanted_num_related_class

        self.initial = models.resnet50(pretrained=True)
        self.initial.fc = torch.nn.Identity()
        self.model_type = 'resnet50'
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

    def set_model_type(self, model_type):
        """
        Sets an initial model on the task that will be used as the architecture
        and initial weights of models created for the task, including taglets and
        the end model
        
        :param initial: the initial model
        """
        self.model_type = model_type

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
