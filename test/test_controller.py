#from .util import check_test_data
from taglets.controller import Controller
from taglets.task import Task
import unittest

import io
import os
import requests
import zipfile


def check_test_data():
    if not os.path.isdir("mnist_sample"):
        r = requests.get("http://cs.brown.edu/people/sbach/mnist_sample.zip")
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall()


class TestController(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        check_test_data()

    def test_mnist(self):
        # Creates task
        task = Task("mnist")
        task.classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        task.unlabeled_image_path = "mnist_sample/train"
        task.evaluation_image_path = "mnist_sample/test"
        task.number_of_channels = 1


        # Executes task
        controller = Controller(use_gpu=False)
        end_model = controller.train_end_model(task)


if __name__ == "__main__":
    unittest.main()
