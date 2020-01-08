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

        self.dataset_type = 'image_classification'

        # ##### the data we will be evaluated based on. We submit our prediciton on these data
        self.test_imgs = [f.name for f in Path("/Users/markh/Downloads/mnist/mnist_sample/test").iterdir() if f.is_file()]
        self.unlabeled_images = 'path to unlabeled imges'
        self.labeled_images = '/datasets/lwll_datasets/mnist/mnist_sample/train',



        def workflow():
            MNIST_module = BaseModule()
            taglets = MNIST_module.get_taglets()
            taglet_executer = TagletExecuter(taglets)
            taglet_executer.train(self.labeled_images)
            label_matrix = taglet_executer.execute(self.unlabeled_images)
            # soft_labels = LabelModel.annotate(label_matrix)
            # end_model = end_model(soft_labels, self.unlabeled_images)
            # [test_predictions] = end_model.prediction(end_model, self.test_images)





