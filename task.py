import numpy as np
from pathlib import Path


class Task:
    """ Task class
    Elaheh: I think there should be a json file for task, we load the json file, and create Task Object. All of the
    information related to Task is in json file
    """
    def __init__(self):
        self.description = ''
        self.target_concepts = []
        self.validation_data = ''
        self.test_data = ''
        self.allowed_datasets = []

class MNIST(Task):
    def __init__(self):
        super().__init__()
        self.classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        self.labels =  [{'id': '56847.png', 'label': '2'},
                          {'id': '45781.png', 'label': '3'},
                          {'id': '40214.png', 'label': '7'},
                          {'id': '49851.png', 'label': '8'},
                          {'id': '46024.png', 'label': '6'},
                          {'id': '13748.png', 'label': '1'},
                          {'id': '13247.png', 'label': '9'},
                          {'id': '39791.png', 'label': '4'},
                          {'id': '37059.png', 'label': '0'},
                          {'id': '46244.png', 'label': '5'}]
        self.data_url = '/datasets/lwll_datasets/mnist/mnist_sample/train',

        self.dataset_type = 'image_classification'

        self.test_imgs = [f.name for f in Path("/Users/markh/Downloads/mnist/mnist_sample/test").iterdir() if f.is_file()]




