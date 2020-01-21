import numpy as np
from pathlib import Path
from PIL import Image
import os
import torch

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

    def add_labeled_images(self, new_labeled_images):
        self.labeled_images.extend(new_labeled_images)

    def get_unlabeled_images(self):
        """
        read, normalize, and upsample images
        :param dir: str, directory path
        :return: 4-d tensor of the images
        """
        raise NotImplementedError()
