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
from torchvision.datasets import CIFAR100
from torchvision.models.resnet import ResNet, BasicBlock
import unittest
import torchvision.models as models

TEST_DATA = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_data/scads")
DB_PATH = os.path.join(TEST_DATA, "test_scads.db")
EMBEDDING_PATH = os.path.join(TEST_DATA, "test_embedding.h5")
CONCEPTNET_PATH = os.path.join(TEST_DATA, "conceptnet")
MNIST_PATH = "mnist"

cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)

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
    def test_cifar_100(self):
        classes = ['bicycle', 'baby', 'oak_tree', 'apple', 'seal', 'beetle', 'plain', 'whale',
                   'ray', 'worm', 'streetcar', 'forest', 'bowl', 'lizard', 'motorcycle', 'man',
                   'fox', 'aquarium_fish', 'bottle', 'palm_tree', 'lion', 'squirrel', 'mouse',
                   'clock', 'train', 'butterfly', 'tiger', 'raccoon', 'bear', 'chair', 'lamp',
                   'turtle', 'rocket', 'table', 'woman', 'otter', 'sunflower', 'orchid', 'girl',
                   'porcupine', 'poppy', 'snake', 'pickup_truck', 'tulip', 'bed', 'house',
                   'sweet_pepper', 'leopard', 'possum', 'flatfish', 'pear', 'shark', 'beaver',
                   'trout', 'cockroach', 'telephone', 'camel', 'crocodile', 'dinosaur', 'bee',
                   'snail', 'tank', 'cattle', 'maple_tree', 'bus', 'hamster', 'mushroom', 'cup',
                   'lawn_mower', 'mountain', 'pine_tree', 'road', 'skunk', 'spider', 'cloud',
                   'couch', 'caterpillar', 'chimpanzee', 'rabbit', 'keyboard', 'skyscraper',
                   'castle', 'crab', 'television', 'willow_tree', 'kangaroo', 'sea', 'wardrobe',
                   'boy', 'can', 'orange', 'dolphin', 'plate', 'lobster', 'elephant', 'wolf',
                   'tractor', 'shrew', 'bridge', 'rose']

        for i in range(len(classes)):
            classes[i] = "/c/en/" + classes[i]

        preprocess = transforms.Compose(
            [transforms.RandomHorizontalFlip(),
             transforms.RandomCrop(size=32,
                                   padding=int(32 * 0.125),
                                   padding_mode='reflect'),
             transforms.ToTensor(),
             transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

        cifar100 = CIFAR100('.', train=True, transform=preprocess, download=True)
        size = len(cifar100)

        labels = []
        for i in range(len(cifar100)):
            labels.append(cifar100[i][1])

        label_per_class = 2500 // 100
        #val_per_class = 5000 // 100
        labels = np.array(labels)
        labeled_idx = []
        validation_idx = []
        unlabeled_idx = np.array(range(len(labels)))
        for i in range(100):
            idx = np.where(labels == i)[0]
            tidx = np.random.choice(idx, label_per_class, False)
            # print
            # labels = np.setdiff1d(labels, tidx)

            # vidx = np.random.choice(np.where(labels == i)[0], val_per_class)
            # labels = np.setdiff1d(labels, vidx)
            # print(len(labels))
            # print(a)
            # assert a > len(idx)

            # validation_idx.extend(vidx)
            labeled_idx.extend(tidx)
        labeled_idx = np.array(labeled_idx)
        validation_idx = np.array(validation_idx)
        assert len(labeled_idx) == 2500

        # maybe find a better way to split based on classes?
        labeled_size = int(size * .20)
        val_size = int(size * .10)

        labeled = Subset(cifar100, labeled_idx)
        val = Subset(cifar100, validation_idx)
        unlabeled = HiddenLabelDataset(Subset(cifar100, unlabeled_idx))

        task = Task("fixmatch-cifar100-test", classes, (32, 32), labeled, unlabeled, None)
        task.set_initial_model(models.resnet18(pretrained=True))
        # Executes task
        controller = UnreliableController(task)
        end_model = controller.train_end_model()

        # Evaluates end model
        cifar100_test = CIFAR100('.', train=False, transform=preprocess, download=True)
        # mnist_test = Subset(mnist_test, [i for i in range(1000)])
        e = end_model.evaluate(cifar100_test)
        print(e)
        self.assertGreater(e, .70)

    def test_mnist_with_scads(self):
        pass

    @classmethod
    def tearDownClass(cls):
        pass


if __name__ == "__main__":
    unittest.main()