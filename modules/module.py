import numpy as np
import task
from taglet import ResnetTaglet, LogisticRegressionTaglet, PrototypeTaglet


class BaseModule:
    """
    Base class for constructing modules.
    """
    def __init__(self, task):
        """
        :param task: The task for the module to consume
        """
        self.task = task

    def get_taglets(self):
        """
        Jeff: We could also have a method to load taglets before returning them here.
        :return: List of taglets
        """
        # for now this is hard-coded
        taglets = [ResnetTaglet(), LogisticRegressionTaglet(), PrototypeTaglet()]
        labeled_data = None     # TODO
        for taglet in taglets:
            taglet.train(labeled_data)
        return taglets
