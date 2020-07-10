class Module:
    """
    Base class for a taglet-creating module. Trains and returns taglets.
    """
    def __init__(self, task):
        """
        Create a new Module.
        :param task: The current task
        """
        self.task = task
        self.taglets = []   # List of taglets must be defined in subclasses

    def train_taglets(self, train_data_loader, val_data_loader, use_gpu):
        """
        Train the Module's taglets.
        :param train_data_loader: A data loader for training data
        :param val_data_loader: A data loader for validation data
        :param use_gpu: Whether or not to use the GPU
        :return: None
        """
        for taglet in self.taglets:
            taglet.train(train_data_loader, val_data_loader, use_gpu)

    def get_taglets(self):
        """
        Return the module's taglets.
        :return: A list of taglets
        """
        return self.taglets
