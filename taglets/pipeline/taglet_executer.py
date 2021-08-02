import numpy as np


class TagletExecutor:
    """
    A class to execute Taglets on unlabeled images.
    """
    def __init__(self):
        """
        Create a new TagletExecutor.
        """
        self.taglets = []

    def set_taglets(self, taglets):
        self.taglets = taglets

    def execute(self, unlabeled_images):
        """
        Execute a list of Taglets and get a label matrix.
        :param unlabeled_images: A dataloader containing unlabeled_images
        :return: A label matrix of size (num_images, num_taglets)
        """
        label_matrix = []
        for taglet in self.taglets:
            labels = taglet.execute(unlabeled_images)
            label_matrix.append(np.expand_dims(labels, 0))
        return np.concatenate(label_matrix, 0)
