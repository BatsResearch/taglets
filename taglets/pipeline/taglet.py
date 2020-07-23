import logging
from .trainable import Trainable
import numpy as np

log = logging.getLogger(__name__)

class Taglet(Trainable):
    """
    A trainable model that produces votes for unlabeled images
    """
    def execute(self, unlabeled_data, use_gpu):
        """
        Execute the Taglet on unlabeled images.

        :param unlabeled_data: A Dataset containing unlabeled data
        :param use_gpu: Whether or not the use the GPU
        :return: A 1-d NumPy array of predicted labels
        """
        outputs = self.predict(unlabeled_data, use_gpu)
        log.info(outputs)
        return np.argmax(outputs, 1)
