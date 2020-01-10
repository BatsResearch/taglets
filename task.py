import numpy as np
from pathlib import Path
from modules.module import BaseModule
from taglet_executer import TagletExecuter


class Task:
    """ Task class
    Elaheh: I think there should be a json file for task, we load the json file, and create Task Object. All of the
    information related to Task is in json file
    """
    def __init__(self):
        self.description = ''
        self.dataset_type = ''
        self.classes = []
        self.test_images = ''
        self.unlabeled_images = ''
        self.labeled_images = ''


class MNIST(Task):
    def __init__(self):
        super().__init__()

        self.description = 'digit recognition'
        self.dataset_type = 'image_classification'
        self.classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        # test_images are the ones that we will be evaluated on. We submit our predictions on these data
        self.test_images = "path to test images"
        self.unlabeled_images = 'path to unlabeled images'
        self.labeled_images = 'path to labeled images'
