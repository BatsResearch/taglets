import numpy as np
import os
from taglets.modules import FineTuneModule
from taglets.pipeline.taglet_executer import TagletExecutor
from taglets.task import Task
import torch
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.models.resnet import ResNet, BasicBlock
import unittest

TEST_DATA = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_data/scads")
DB_PATH = os.path.join(TEST_DATA, "test_scads.db")
CONCEPTNET_PATH = os.path.join(TEST_DATA, "conceptnet")
MNIST_PATH = "mnist"


class HiddenLabelDataset(Dataset):
    """
    Wraps a labeled dataset so that it appears unlabeled
    """
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        return img

    def __len__(self):
        return len(self.dataset)


class MnistResNet(ResNet):
    """
    A small ResNet for MNIST.
    """
    def __init__(self):
        """
        Create a new MnistResNet model.
        """
        super(MnistResNet, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes=10)
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.fc = torch.nn.Identity()


class TestTagletExecuter(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Creates task
        classes = ['/c/en/zero',
                   '/c/en/one',
                   '/c/en/two',
                   '/c/en/three',
                   '/c/en/four',
                   '/c/en/five',
                   '/c/en/six',
                   '/c/en/seven',
                   '/c/en/eight',
                   '/c/en/nine']

        preprocess = transforms.Compose(
            [transforms.Grayscale(num_output_channels=3),
             transforms.ToTensor()])

        mnist = MNIST('.', train=True, transform=preprocess, download=True)
        size = 5
        labeled = Subset(mnist, [i for i in range(size)])
        cls.unlabeled = HiddenLabelDataset(Subset(mnist, [i for i in range(size, 2 * size)]))
        val = Subset(mnist, [i for i in range(2 * size, 3 * size)])
        task = Task("mnist-test", classes, (28, 28), labeled, cls.unlabeled, val)
        task.set_initial_model(MnistResNet())

        # Train and get Taglets
        module = FineTuneModule(task)
        module.train_taglets(labeled, val)
        cls.taglets = module.get_taglets()

        # Execute Taglets
        executor = TagletExecutor()
        executor.set_taglets(cls.taglets)
        cls.label_matrix = executor.execute(cls.unlabeled)

    def test_weak_label_shape(self):
        self.assertTrue(self.label_matrix.shape[0] == len(self.unlabeled))
        self.assertTrue(self.label_matrix.shape[1] == len(self.taglets))

    def test_weak_label_correctness(self):
        taglet_output = self.taglets[0].execute(self.unlabeled)
        self.assertTrue(np.array_equal(taglet_output, self.label_matrix[:, 0]))


if __name__ == "__main__":
    unittest.main()
