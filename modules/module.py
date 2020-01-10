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
            taglet.train(labeled_images)

    def get_taglets(self):
        """
        :return: List of taglets
        """
        return self.taglets
