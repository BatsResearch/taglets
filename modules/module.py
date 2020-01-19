import numpy as np
from pathlib import Path
from random import sample
from taglet import *
# from taglet import ResnetTaglet, LogisticRegressionTaglet, PrototypeTaglet


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

    def train_taglets(self):
        """call train method for all of taglets in this module"""
        for taglet in self.taglets:
            taglet.train()

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

        self.taglets = [FineTuenTaglet(task)]    # TODO: add Transfer, MTLTaglet in taglet.py


class ActiveLearningModule:
    """
    Active Learning module
    """

    def __init__(self, task, available_budget):
        """
        :param task: current task
        :param available_budget: maximum number of candidates we could choose for labeling
        """
        self.task = task
        self.available_budget = available_budget
        self.candidates = [] # List of candidates to be labeled

    def find_candidates(self):
        """select a set of candidates to be labeled"""
        self.candidates = []


class LeastConfidenceActiveLearning(ActiveLearningModule):
    """
    An active learning Module that chose the candidate set based on lowest confidence score.
    """

    def __init__(self, task, available_budget):
        super().__init__(task, available_budget)

    def find_candidates(self):
        """return a list of candidates using confidence score"""
        raise NotImplementedError


class RandomActiveLearning(ActiveLearningModule):
    """
    An active learning Module that chose the candidate set randomly.
    """

    def __init__(self, task, available_budget):
        super().__init__(task, available_budget)

    def find_candidates(self):
        """select a random set of candidates to be labeled"""

        image_dir = self.task.unlabeled_image_path
        print((image_dir))
        unlabeled_imgs = [f.name for f in Path(image_dir).iterdir() if f.is_file()]

        self.candidates = sample(unlabeled_imgs, self.available_budget)


