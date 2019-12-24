import numpy as np
import task


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
        raise NotImplementedError()