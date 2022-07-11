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

    def train_taglets(self, train_data, val_data, unlabeled_data=None, num_checkpoint=None):
        """
        Train the Module's taglets.
        :param train_data: A Torch Dataset of training data
        :param val_data: A Torch Dataset of validation data
        :param unlabeled_data: A Torch Dataset of unlabeled data
        :return: None
        """
        for taglet in self.taglets:
            taglet.train(train_data, val_data, unlabeled_data, num_checkpoint=num_checkpoint)
            
    def get_valid_taglets(self):
        """
        Return the module's taglets.
        :return: A list of taglets
        """
        valid_taglets = []
        for taglet in self.taglets:
            if taglet.valid:
                valid_taglets.append(taglet)
        return valid_taglets

    def get_taglets(self):
        """
        Return the module's taglets.
        :return: A list of taglets
        """
        return self.taglets
