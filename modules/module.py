import numpy as np
from pathlib import Path
from random import sample
from taglet import *


class BaseModule:
    """
    Base class for constructing modules.
    """

    def __init__(self, task):
        """
        :param task: The task for the module to consume
        """
        self.task = task
        self.taglets = []   # List of taglets must be defined in subclasses

    def train_taglets(self, train_data_loader, val_data_loader, test_data_loader):
        """call train method for all of taglets in this module"""
        for taglet in self.taglets:
            taglet.train(train_data_loader, val_data_loader, test_data_loader)

    def get_taglets(self):
        """
        :return: List of taglets
        """
        return self.taglets


class TransferModule(BaseModule):
    """
    Module related to transfer learning when we have enough amount of labeled data for fine tuning
    """

    def __init__(self, task):
        super().__init__(task)
        self.taglets = [FineTuneTaglet(task)]    # TODO: add Transfer, MTLTaglet in taglet.py


class ActiveLearningModule:
    """
    Active Learning module
    """

    def __init__(self, task):
        """
        :param task: current task
        :param available_budget: maximum number of candidates we could choose for labeling
        """
        self.task = task
        self.labeled = set()    # List of candidates already labeled

    def find_candidates(self, available_budget):
        """select a set of candidates to be labeled"""
        return []


class LeastConfidenceActiveLearning(ActiveLearningModule):
    """
    An active learning Module that chose the candidate set based on lowest confidence score.
    """

    def __init__(self, task):
        super().__init__(task)

    def find_candidates(self, available_budget):
        """return a list of candidates using confidence score"""
        raise NotImplementedError


class RandomActiveLearning(ActiveLearningModule):
    """
    An active learning Module that chose the candidate set randomly.
    """

    def __init__(self, task):
        super().__init__(task)

    def find_candidates(self, available_budget):
        """select a random set of candidates to be labeled"""

        image_dir = self.task.unlabeled_image_path
        unlabeled_images = [f.name for f in Path(image_dir).iterdir() if f.is_file() and f.name not in self.labeled]
        to_request = sample(unlabeled_images, available_budget)
        self.labeled.update(to_request)
        return to_request
