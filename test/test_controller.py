from taglets.controller import Controller
from taglets.modules.module import Module
from taglets.scads import Scads
from taglets.scads.create.install import Installer, MnistInstallation
from taglets.task import Task

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.models.resnet import ResNet, BasicBlock
import unittest


MNIST.resources = [
    ('https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz', 'f68b3c2dcbeaaa9fbdd348bbdeb94873'),
    ('https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz', 'd53e105ee54ea40749a09fcbcd1e9432'),
    ('https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz', '9fb629c4189551a2d022fa330f9573f3'),
    ('https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz', 'ec29112dd5afa0611ce80d1b7f02629c')
]


TEST_DATA = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_data/scads")
DB_PATH = os.path.join(TEST_DATA, "test_scads.db")
EMBEDDING_PATH = os.path.join(TEST_DATA, "test_embedding.h5")
CONCEPTNET_PATH = os.path.join(TEST_DATA, "conceptnet")
MNIST_PATH = "mnist"


class HiddenLabelDataset(Dataset):
    """
    Wraps a labeled dataset so that it appears unlabeled
    """
    def __init__(self, dataset):
        self.subset = dataset
        self.dataset = self.subset.dataset

    def __getitem__(self, idx):
        data = self.subset[idx]
        try:
            img1, img2, _ = data
            return img1, img2

        except ValueError:
            return data[0]

    def __len__(self):
        return len(self.subset)


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


class BadModule1(Module):
    def __init__(self, task):
        super(BadModule1, self).__init__(task)
        raise Exception("Deliberate error.")


class BadModule2(Module):
    def train_taglets(self, train_data, val_data):
        raise Exception("Deliberate error.")


class UnreliableController(Controller):
    def _get_taglets_modules(self):
        modules = super(UnreliableController, self)._get_taglets_modules()
        modules.append(BadModule1)
        modules.append(BadModule2)
        return modules


class TestController(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Set Up Scads
        if os.path.isfile(DB_PATH):
            os.remove(DB_PATH)
        installer = Installer(DB_PATH)
        installer.install_conceptnet(CONCEPTNET_PATH)
        installer.install_dataset(TEST_DATA, MNIST_PATH, MnistInstallation())

        # Build ScadsEmbedding File
        arr = []
        for i in range(10):
            l = [0.0] * 10
            l[i] = 1.0
            arr.append(l)
        arr = np.asarray(arr)
        label_list = ['/c/en/zero',
                      '/c/en/one',
                      '/c/en/two',
                      '/c/en/three',
                      '/c/en/four',
                      '/c/en/five',
                      '/c/en/six',
                      '/c/en/seven',
                      '/c/en/eight',
                      '/c/en/nine']
        df = pd.DataFrame(arr, index=label_list, dtype='f')
        df.to_hdf(EMBEDDING_PATH, key='mat', mode='w')

    def test_mnist(self):
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
        size = int(len(mnist) / 50)
        labeled = Subset(mnist, [i for i in range(size)])

        # this is necessary because Fixmatch overrides the MNIST transform attribute
        unlabeled = HiddenLabelDataset(Subset(mnist, [i for i in range(size, 2 * size)]))
        val = Subset(mnist, [i for i in range(2 * size, 3 * size)])
        task = Task("mnist-test", classes, (28, 28), labeled, unlabeled, val)
        task.set_initial_model(MnistResNet())

        # Executes task
        controller = UnreliableController(task)
        end_model = controller.train_end_model()

        # Evaluates end model
        mnist_test = MNIST('.', train=False, transform=preprocess, download=True)
        mnist_test = Subset(mnist_test, [i for i in range(1000)])
        self.assertGreater(end_model.evaluate(mnist_test), .85)

    def test_mnist_with_scads(self):
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
        size = int(len(mnist) / 50)
        labeled = Subset(mnist, [i for i in range(size)])
        unlabeled = HiddenLabelDataset(Subset(mnist, [i for i in range(size, 2 * size)]))
        val = Subset(mnist, [i for i in range(2 * size, 3 * size)])
        task = Task(
            "mnist-test", classes, (28, 28), labeled, unlabeled, val, scads_path=DB_PATH,
            scads_embedding_path=EMBEDDING_PATH
        )
        task.set_initial_model(MnistResNet())
        Scads.set_root_path(TEST_DATA)

        # Executes task
        controller = Controller(task)
        end_model = controller.train_end_model()

        # Evaluates end model
        mnist_test = MNIST('.', train=False, transform=preprocess, download=True)
        mnist_test = Subset(mnist_test, [i for i in range(1000)])
        self.assertGreater(end_model.evaluate(mnist_test), .9)

    @classmethod
    def tearDownClass(cls):
        os.remove(DB_PATH)


if __name__ == "__main__":
    unittest.main()
