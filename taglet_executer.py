import numpy as np


class TagletExecuter:
    """
    TagletExecuter class (Equivalent to LFApplier)
    """
    def __init__(self, taglets):
        self.taglets = taglets

    def execute(self, images):
        """
        Perform execute function of all Taglets
        :return: A label matrix of size (num_images, num_taglets)
        """
        raise NotImplementedError()
