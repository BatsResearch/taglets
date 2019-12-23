import numpy as np
import Task


class BaseModule:
    """ Base class for constructing modules.

    """
    def __init__(self, task):
        self.task = task
        print('base class')
