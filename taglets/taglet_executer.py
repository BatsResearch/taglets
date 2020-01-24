import numpy as np


class TagletExecutor:
    """
    A class to execute Taglets on unlabeled images.
    """
    def __init__(self, taglets):
        """
        Create a new TagletExecutor.
        :param taglets: The list of Taglets to execute
        """
        self.taglets = taglets

    def execute(self, unlabeled_images, use_gpu):
        """
        Execute a list of Taglets and get a label matrix.
        :param unlabeled_images: A dataloader containing unlabeled_images
        :param use_gpu: Whether or not to use the GPU
        :return: A label matrix of size (num_images, num_taglets)
        """
        num_images = len(unlabeled_images.dataset)
        num_taglets = len(self.taglets)
        label_matrix = np.zeros((num_images, num_taglets))
        for i in range(num_taglets):
            label_matrix[:, i] = self.taglets[i].execute(unlabeled_images, use_gpu)

        return label_matrix
