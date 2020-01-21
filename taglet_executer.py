import numpy as np


class TagletExecuter:
    """
    TagletExecuter class (Equivalent to LFApplier)
    """
    def __init__(self, taglets):
        self.taglets = taglets

    def execute(self, unlabeled_images):
        """
        Perform execute function of all Taglets
        :param unlabeled_images: unlabeled_images
        :param use_gpu: a boolean indicating if the taglets should execute on gpu
        :return: A label matrix of size (num_images, num_taglets)
        """
        num_images = len(unlabeled_images.dataset)
        num_taglets = len(self.taglets)
        label_matrix = np.zeros((num_images, num_taglets))

        for i in range(num_taglets):
            label_matrix[:, i] = self.taglets[i].execute(unlabeled_images)

        return label_matrix
