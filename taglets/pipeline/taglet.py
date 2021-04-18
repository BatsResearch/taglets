import logging
from .trainable import ImageTrainable, VideoTrainable
import numpy as np

log = logging.getLogger(__name__)

class TrainableMixin:
    def execute(self, unlabeled_data):
        """
        Execute the Taglet on unlabeled images.

        :param unlabeled_data: A Dataset containing unlabeled data
        :return: A 1-d NumPy array of predicted labels
        """
        outputs = self.predict(unlabeled_data)
        return np.argmax(outputs, 1)


class ImageTaglet(ImageTrainable, TrainableMixin):
    """
    A trainable model that produces votes for unlabeled images
    """


class VideoTaglet(VideoTrainable, TrainableMixin):
    """
    A trainable model that produces votes for unlabeled videos
    """

