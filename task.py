import numpy as np
from pathlib import Path
from modules.module import BaseModule
from taglet_executer import TagletExecuter


class Task:
    """ Task class """

    def __init__(self, metadata):
        self.description = ''
        self.adaptation_can_use_pretrained_model = metadata.adaptation_can_use_pretrained_model
        self.adaptation_dataset = metadata.adaptation_dataset
        self.adaptation_evaluation_metrics = metadata.adaptation_evaluation_metrics
        self.adaptation_label_budget = metadata.adaptation_label_budget
        self.base_can_use_pretrained_model = metadata.base_can_use_pretrained_model
        self.base_dataset = metadata.base_dataset
        self.base_evaluation_metrics = metadata.base_evaluation_metrics
        self.base_label_budget = metadata.base_label_budget
        self.problem_type = metadata.problem_type
        self.task_id = metadata.metadata

        self.classes = []
        self.test_images = "path to test images"
        self.unlabeled_images = 'path to unlabeled images'
        self.labeled_images = 'path to labeled images'

