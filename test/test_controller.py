from taglets.controller import Controller
from taglets.task import Task

import unittest

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.models.resnet import ResNet, BasicBlock


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


class TestController(unittest.TestCase):
    def test_mnist(self):
        # Creates task
        classes = {
            0: '/c/en/zero/n/wn/quantity',
            1: '/c/en/one/n/wn/quantity',
            2: '/c/en/two/n/wn/quantity',
            3: '/c/en/three/n/wn/quantity',
            4: '/c/en/four/n/wn/quantity',
            5: '/c/en/five/n/wn/quantity',
            6: '/c/en/six/n/wn/quantity',
            7: '/c/en/seven/n/wn/quantity',
            8: '/c/en/eight/n/wn/quantity',
            9: '/c/en/nine/n/wn/quantity',
        }

        preprocess = transforms.Compose(
            [transforms.Grayscale(num_output_channels=3),
             transforms.ToTensor()])

        mnist = MNIST('.', train=True, transform=preprocess, download=True)
        size = int(len(mnist) / 50)
        labeled = Subset(mnist, [i for i in range(size)])
        unlabeled = HiddenLabelDataset(Subset(mnist, [i for i in range(size, 2 * size)]))
        val = Subset(mnist, [i for i in range(2 * size, 3 * size)])
        task = Task("mnist-test", classes, (28, 28), labeled, unlabeled, val)
        task.set_initial_model(MnistResNet())

        # Executes task
        controller = Controller(task, use_gpu=False)
        end_model = controller.train_end_model()

        # Evaluates end model
        mnist_test = MNIST('.', train=False, transform=preprocess, download=True)
        mnist_test = Subset(mnist_test, [i for i in range(1000)])
        mnist_test = DataLoader(mnist_test, batch_size=32)
        self.assertGreater(end_model.evaluate(mnist_test, use_gpu=False), .9)


if __name__ == "__main__":
    unittest.main()
