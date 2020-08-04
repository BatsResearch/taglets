from taglets.controller import Controller
from taglets.scads import Scads
from taglets.scads.create.install import Installer, MnistInstallation
from taglets.task import Task

import os
import unittest
import multiprocessing as mp

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


class LabeledSubset(Dataset):
    def __init__(self, dataset, labels, indices):
        self.dataset = dataset
        self.labels = labels[indices]
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


class TestController(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Set Up Scads
        installer = Installer(DB_PATH)
        installer.install_conceptnet(CONCEPTNET_PATH)
        installer.install_dataset(TEST_DATA, MNIST_PATH, MnistInstallation())

    def test_mnist(self):
        # Creates task
        classes = ['/c/en/zero/n/wn/quantity',
                   '/c/en/one/n/wn/quantity',
                   '/c/en/two/n/wn/quantity',
                   '/c/en/three/n/wn/quantity',
                   '/c/en/four/n/wn/quantity',
                   '/c/en/five/n/wn/quantity',
                   '/c/en/six/n/wn/quantity',
                   '/c/en/seven/n/wn/quantity',
                   '/c/en/eight/n/wn/quantity',
                   '/c/en/nine/n/wn/quantity']

        preprocess = transforms.Compose(
            [transforms.Grayscale(num_output_channels=3),
             transforms.ToTensor()])

        mnist = MNIST('.', train=True, transform=preprocess, download=True)
        size = int(len(mnist) / 50)
        labeled = LabeledSubset(mnist, mnist.targets, [i for i in range(size)])
        unlabeled = HiddenLabelDataset(Subset(mnist, [i for i in range(size, 2 * size)]))
        val = LabeledSubset(mnist, mnist.targets, [i for i in range(2 * size, 3 * size)])
        task = Task("mnist-test", classes, (28, 28), labeled, unlabeled, val)
        task.set_initial_model(MnistResNet())

        # Executes task
        controller = Controller(task, use_gpu=False)
        end_model = controller.train_end_model()

        # Evaluates end model
        mnist_test = MNIST('.', train=False, transform=preprocess, download=True)
        mnist_test = Subset(mnist_test, [i for i in range(1000)])
        self.assertGreater(end_model.evaluate(mnist_test, use_gpu=False), .9)

    def test_mnist_with_scads(self):
        # Creates task
        classes = ['/c/en/zero/n/wn/quantity',
                   '/c/en/one/n/wn/quantity',
                   '/c/en/two/n/wn/quantity',
                   '/c/en/three/n/wn/quantity',
                   '/c/en/four/n/wn/quantity',
                   '/c/en/five/n/wn/quantity',
                   '/c/en/six/n/wn/quantity',
                   '/c/en/seven/n/wn/quantity',
                   '/c/en/eight/n/wn/quantity',
                   '/c/en/nine/n/wn/quantity']

        preprocess = transforms.Compose(
            [transforms.Grayscale(num_output_channels=3),
             transforms.ToTensor()])

        mnist = MNIST('.', train=True, transform=preprocess, download=True)
        size = int(len(mnist) / 50)
        labeled = LabeledSubset(mnist, mnist.targets, [i for i in range(size)])
        unlabeled = HiddenLabelDataset(LabeledSubset(mnist, mnist.targets, [i for i in range(size, 2 * size)]))
        val = LabeledSubset(mnist, mnist.targets, [i for i in range(2 * size, 3 * size)])
        task = Task(
            "mnist-test", classes, (28, 28), labeled, unlabeled, val, scads_path=DB_PATH
        )
        task.set_initial_model(MnistResNet())
        Scads.set_root_path(TEST_DATA)

        # Executes task
        controller = Controller(task, use_gpu=False)
        end_model = controller.train_end_model()

        # Evaluates end model
        mnist_test = MNIST('.', train=False, transform=preprocess, download=True)
        mnist_test = Subset(mnist_test, [i for i in range(1000)])
        self.assertGreater(end_model.evaluate(mnist_test, use_gpu=False), .9)

    @classmethod
    def tearDownClass(cls):
        os.remove(DB_PATH)


if __name__ == "__main__":
    unittest.main()
