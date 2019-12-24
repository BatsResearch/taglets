import numpy as np


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

