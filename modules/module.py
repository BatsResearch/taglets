import numpy as np
import task
from taglet import ResnetTaglet, LogisticRegressionTaglet, PrototypeTaglet


class BaseModule:
    """
    Base class for constructing modules.
    """

    def __init__(self, task):
        """
        Jeff: The list of taglets will look different based future subclasses of this BaseModule
        :param task: The task for the module to consume
        """
        self.task = task
        # Currently hardcoded
        self.taglets = [ResnetTaglet(), LogisticRegressionTaglet(), PrototypeTaglet()]

    def train_taglets(self, labeled_images, batch_size=64, use_gpu=True):
        for taglet in self.taglets:
            taglet.train(labeled_images, batch_size, use_gpu)

    def get_taglets(self):
        """
        :return: List of taglets
        """
        return self.taglets


class TransferModul(BaseModule):
    """
    Module related to transfer learning when we have enough amount of labled data for fine tuning
    """

    def __init__(self):
        super().__init__()
        self.taglets = [MTLTaglet(), ResnetTaglet(), LogisticRegressionTaglet()]#TODO add MTLTaglet in tagelet.py

    def get_taglets(self):
        return self.taglets


