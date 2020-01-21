from taglets.taglet import *


class Module:
    """
    Base class for a module. Trains and returns taglets.
    """
    def __init__(self, task):
        """
        Create a new Module.
        :param task: The current task
        """
        self.task = task
        self.taglets = []   # List of taglets must be defined in subclasses

    def train_taglets(self, train_data_loader, val_data_loader, test_data_loader):
        """
        Train the Module's taglets.
        :param train_data_loader: A data loader for training data
        :param val_data_loader: A data loader for validation data
        :param test_data_loader: A data loader for testing data
        :return: None
        """
        for taglet in self.taglets:
            taglet.train(train_data_loader, val_data_loader, test_data_loader)

    def get_taglets(self):
        """
        Return the module's taglets.
        :return: A list of taglets
        """
        return self.taglets


class TransferModule(Module):
    """
    A module used for transfer learning. Used when there is enough labeled data for fine tuning.
    """
    def __init__(self, task):
        super().__init__(task)
        self.taglets = [FineTuneTaglet(task)]
