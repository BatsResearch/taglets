from pathlib import Path
from random import sample


class ActiveLearningModule:
    """
    Active Learning module
    """

    def __init__(self, task):
        """
        :param task: current task
        :param available_budget: maximum number of candidates we could choose for labeling
        """
        self.task = task

    def find_candidates(self, available_budget):
        """select a set of candidates to be labeled"""
        return []


class LeastConfidenceActiveLearning(ActiveLearningModule):
    """
    An active learning Module that chose the candidate set based on lowest confidence score.
    """

    def __init__(self, task):
        super().__init__(task)

    def find_candidates(self, available_budget):
        """return a list of candidates using confidence score"""
        raise NotImplementedError


class RandomActiveLearning(ActiveLearningModule):
    """
    An active learning Module that chose the candidate set randomly.
    """

    def __init__(self, task):
        super().__init__(task)
        self.labeled = set()  # List of candidates already labeled

    def find_candidates(self, available_budget):
        """select a random set of candidates to be labeled"""

        image_dir = self.task.unlabeled_image_path
        unlabeled_images = [f.name for f in Path(image_dir).iterdir() if f.is_file() and f.name not in self.labeled]
        to_request = sample(unlabeled_images, available_budget)
        self.labeled.update(to_request)
        return to_request
