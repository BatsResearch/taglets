import logging
from .trainable import ImageTrainable, VideoTrainable
import numpy as np

log = logging.getLogger(__name__)

class TagletMixin:
    def execute(self, unlabeled_data):
        """
        Execute the Taglet on unlabeled images.

        :param unlabeled_data: A Dataset containing unlabeled data
        :return: A 1-d NumPy array of predicted labels
        """
        outputs = self.predict(unlabeled_data)
        if self.name == 'svc-video':
            log.info("EXECUTE SVC MODULE PREDICTIONS")
            return outputs
        
        return np.argmax(outputs, 1)


class ImageTaglet(ImageTrainable, TagletMixin):
    """
    A trainable model that produces votes for unlabeled images
    """


class VideoTaglet(VideoTrainable, TagletMixin):
    """
    A trainable model that produces votes for unlabeled videos
    """

