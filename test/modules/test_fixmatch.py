from taglets.modules.fixmatch import FixMatchModule
from taglets.modules.fixmatch import TransformFixMatch
from .test_module import TestModule
import unittest

import os
from taglets.task import Task
from taglets.scads import Scads
from taglets.scads.create.scads_classes import Dataset
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder


TEST_DATA = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../test_data/modules")
DB_PATH = os.path.join(TEST_DATA, "test_module.db")
EMBEDDING_PATH = os.path.join(TEST_DATA, "test_embedding.h5")


class HiddenLabelDataset(Dataset):
    """
    Wraps a labeled dataset so that it appears unlabeled
    """
    def __init__(self, dataset):
        self.subset = dataset
        #self.dataset = self.subset.dataset

    def __getitem__(self, idx):
        data = self.subset[idx]
        try:
            img1, img2, _ = data
            return img1, img2
        except ValueError:
            return data[0]

    def __len__(self):
        return len(self.subset)



class TestFixMatch(TestModule, unittest.TestCase):
    def _get_module(self, task):
        return FixMatchModule(task)

    def setUp(self):
        preprocess = transforms.Compose(
            [transforms.CenterCrop(224),
             transforms.ToTensor()])

        self.train = ImageFolder(os.path.join(TEST_DATA, "train"), transform=preprocess)
        self.val = ImageFolder(os.path.join(TEST_DATA, "val"), transform=preprocess)
        self.test = ImageFolder(os.path.join(TEST_DATA, "test"), transform=preprocess)
        self.unlabeled = ImageFolder(os.path.join(TEST_DATA, "unlabeled"),
                                     transform=transforms.Compose(
                                         [transforms.CenterCrop(224)]))

        self.unlabeled = HiddenLabelDataset(self.unlabeled)
        self.task = Task("test_module", ["/c/en/airplane", "/c/en/cat", "/c/en/dog"],
                         (224, 224), self.train, self.unlabeled, self.val,
                         scads_path=DB_PATH, scads_embedding_path=EMBEDDING_PATH)
        Scads.set_root_path(TEST_DATA)
        torch.manual_seed(0)


if __name__ == '__main__':
    unittest.main()
