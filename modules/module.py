import numpy as np
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
        self.taglets = []   # List of taglets must be defined in subclasses

    def train_taglets(self, labeled_images, lr=1e-3, num_epochs=100, batch_size=64, use_gpu=True):
        
        # TODO: seperate labeled_images to images and labels
        
        for taglet in self.taglets:
            taglet.train(images, labels, lr=lr, num_epochs=num_epochs, batch_size=batch_size, use_gpu=use_gpu)

    def get_taglets(self):
        """
        :return: List of taglets
        """
        return self.taglets


class TransferModule(BaseModule):
    """
    Module related to transfer learning when we have enough amount of labeled data for fine tuning
    """

    def __init__(self):
        super().__init__()
        self.taglets = [MTLTaglet(), ResnetTaglet(), LogisticRegressionTaglet()]    # TODO: add MTLTaglet in taglet.py
