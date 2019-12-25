import numpy as np


class Taglet:
    """
    Taglet class
    """
    def __init__(self):
        raise NotImplementedError()

    def execute(self, images, use_gpu=True):
        """
        Top: I add use_gpu as another argument for this function.
        Execute the taglet on a batch of images.
        :return: A batch of labels
        """
        raise NotImplementedError()
