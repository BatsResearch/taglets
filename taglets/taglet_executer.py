import numpy as np


class TagletExecutor:
    """
    A class to execute Taglets on unlabeled images.
    """
    def __init__(self):
        """
        Create a new TagletExecutor.
        :param taglets: The list of Taglets to execute
        """
        self.taglets = []

    def set_taglets(self, taglets):
        self.taglets = taglets

    def execute(self, unlabeled_images, use_gpu, testing):
        """
        Execute a list of Taglets and get a label matrix.
        :param unlabeled_images: A dataloader containing unlabeled_images
        :param use_gpu: Whether or not to use the GPU
        :param available_budget: The number of labels to request
        :return: A label matrix of size (num_images, num_taglets)
        """
        label_matrix = []
        probabilities = []
        for taglet in self.taglets:
            if taglet.name == "finetune":
                labels, probabilities = taglet.execute(unlabeled_images, use_gpu, testing)
            else:
                labels = taglet.execute(unlabeled_images, use_gpu, testing)
            label_matrix.append(labels)
        return np.transpose(label_matrix), probabilities
