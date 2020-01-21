from taglets.taglet import *


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
        self.taglets = [FineTuneTaglet(task)]
