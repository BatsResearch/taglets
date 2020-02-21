from taglets.controller import Controller
from taglets.task import Task

import unittest

import logging
import sys
from torch.utils.data import Dataset, Subset
from torchvision.datasets import MNIST


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
        return len(self.da)


class TestController(unittest.TestCase):
    def setUp(self):
        logger = logging.getLogger()
        logger.level = logging.INFO
        stream_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

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

        mnist = MNIST('.', train=True, download=True)
        size = int(len(mnist) / 20)
        labeled = Subset(mnist, [i for i in range(size)])
        unlabeled = HiddenLabelDataset(Subset(mnist, [i for i in range(size, 2 * size)]))
        val = Subset(mnist, [i for i in range(2 * size, 3 * size)])
        task = Task("mnist-test", classes, labeled, unlabeled, val)

        # Executes task
        controller = Controller(use_gpu=False)
        _ = controller.train_end_model(task)


if __name__ == "__main__":
    unittest.main()
