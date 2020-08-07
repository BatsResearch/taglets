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

    def train_taglets(self, train_data, val_data):
        """
        Train the Module's taglets.
        :param train_data: A Torch Dataset of training data
        :param val_data: A Torch Dataset of validation data
        :return: None
        """
        for taglet in self.taglets:
            taglet.train(train_data, val_data)

    def get_taglets(self):
        """
        Return the module's taglets.
        :return: A list of taglets
        """
        return self.taglets
