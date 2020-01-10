import numpy as np
import torchvision.models as models


class Taglet:
    """
    Taglet class
    """
    def __init__(self):
        raise NotImplementedError()

    def execute(self, unlabeled_images, batch_size=64, use_gpu=True):
        """
        Top: I add use_gpu as another argument for this function.
        Execute the taglet on a batch of images.
        :return: A batch of labels
        """
        raise NotImplementedError()


class ResnetTaglet(Taglet):
    def __init__(self):
        super().__init__()
        self.pretrained = True

        self.model = models.resnet18(pretrained=self.pretrained)

    def train(self, labeled_data, batch_size, use_gpu):
        print('train model on labeled data')

    def execute(self, unlabeled_images, use_gpu=True):
        print('execute vote on unlabeled data')


class LogisticRegressionTaglet(Taglet):
    def __init__(self):
        super().__init__()
        self.pretrained = True

        # self.model = one layer NN

    def train(self, labeled_data, batch_size, use_gpu):
        print('train model on labeled data')

    def execute(self, unlabeled_images, use_gpu=True):
        print('execute vote on unlabeled data')


class PrototypeTaglet(Taglet):
    def __init__(self):
        super().__init__()
        self.pretrained = True

        # self.model = Peilin will take care of this

    def train(self, labeled_data, batch_size, use_gpu):
        print('train model on labeled data')

    def execute(self, unlabeled_images, use_gpu=True):
        print('execute vote on unlabeled data')
