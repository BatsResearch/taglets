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

    def train_taglets(self, train_data_loader, val_data_loader, use_gpu, testing):
        """
        Train the Module's taglets.
        :param train_data_loader: A data loader for training data
        :param val_data_loader: A data loader for validation data
        :param test_data_loader: A data loader for testing data
        :param use_gpu: Whether or not to use the GPU
        :return: None
        """
        for taglet in self.taglets:
            taglet.train(train_data_loader, val_data_loader, use_gpu, testing)

    def get_taglets(self):
        """
        Return the module's taglets.
        :return: A list of taglets
        """
        return self.taglets


class FineTuneModule(Module):
    """
    A module used for transfer learning. Used when there is enough labeled data for fine tuning.
    """
    def __init__(self, task):
        super().__init__(task)
        self.taglets = [PrototypeTaglet(task), PrototypeTaglet(task), FineTuneTaglet(task)]


class TransferModule(Module):
    def __init__(self, task):
        super().__init__(task)
        self.taglets = [TransferTaglet(task)]
