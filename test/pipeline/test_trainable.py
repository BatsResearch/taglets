from taglets.pipeline import Trainable
from taglets.task import Task
import unittest
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.models.resnet import ResNet, BasicBlock
from test.test_utils import set_headers


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


class TestTrainable(unittest.TestCase):
    def test_predict(self):
        set_headers()
        # Creates task
        classes = [
            '/c/en/zero',
            '/c/en/one',
            '/c/en/two',
            '/c/en/three',
            '/c/en/four',
            '/c/en/five',
            '/c/en/six',
            '/c/en/seven',
            '/c/en/eight',
            '/c/en/nine',
        ]

        preprocess = transforms.Compose(
            [transforms.Grayscale(num_output_channels=3),
             transforms.ToTensor()])

        mnist = MNIST('.', train=True, transform=preprocess, download=True)
        # Odd number to check imbalance with 2 processes
        size = 25
        labeled = Subset(mnist, [i for i in range(size)])
        unlabeled = HiddenLabelDataset(Subset(mnist, [i for i in range(size, 2 * size)]))
        val = Subset(mnist, [i for i in range(2 * size, 3 * size)])
        task = Task("mnist-test", classes, (28, 28), labeled, unlabeled, val)
        task.set_initial_model(MnistResNet())

        trainable = Trainable(task)
        trainable.model = torch.nn.Sequential(
            trainable.model, torch.nn.Linear(512, 10)
        )
        trainable.n_proc = 1
        trainable.num_epochs = 1
        trainable.train(labeled, val)

        # Checks that single and multiprocess both match a serial implementation
        p1 = trainable.predict(unlabeled)
        trainable.n_proc = 2
        p2 = trainable.predict(unlabeled)
        p3 = serial_predict(trainable.model, unlabeled)

        self.assertLess((abs(p1 - p2)).sum(), 1e-3)
        self.assertLess((abs(p1 - p3)).sum(), 1e-3)


def serial_predict(model, unlabeled_data):
    unlabeled_data_loader = torch.utils.data.DataLoader(
        dataset=unlabeled_data, batch_size=32, shuffle=False
    )

    model.eval()
    model = model.cpu()

    outputs = []
    for inputs in unlabeled_data_loader:
        with torch.set_grad_enabled(False):
            output = model(inputs)
            outputs.append(torch.nn.functional.softmax(output, 1))
    return torch.cat(outputs).detach().numpy()


if __name__ == "__main__":
    unittest.main()
