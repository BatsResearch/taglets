import numpy as np
import torchvision.models as models


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

class resnet_taglet(Taglet):
    def __init__(self):
        super().__init__()
        self.pretrained = True

        self.model = models.resnet18(pretrained=self.pretrained)


    def train(self, labeled_data):
        print('train model on labeled data')

    def execute(self, images, use_gpu=True):
        print('execute vote on test data')



