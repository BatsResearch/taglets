import numpy as np
from pathlib import Path
from modules.module import BaseModule
from taglet_executer import TagletExecuter


class Task:
    """ Task class """

    def __init__(self, metadata):
        self.description = ''
        self.problem_type = metadata['problem_type']
        self.task_id = metadata['task_id']
        self.classes = []
        self.evaluation_image_path = "path to test images"
        self.unlabeled_image_path = 'path to unlabeled images'
        self.labeled_images = []    # A list of tuples with name and label e.g., ['1.png', '2'], ['2.png', '7'], etc.
        self.number_of_channels = None
