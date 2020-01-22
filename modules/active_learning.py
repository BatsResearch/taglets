from pathlib import Path
from random import sample


class ActiveLearningModule:
    """
    The base class for an active learning module. Used to find examples to label.
    """

    def __init__(self, task):
        """
        Create a new ActiveLearningModule.
        :param task: The current task
        """
        self.task = task

    def find_candidates(self, available_budget):
        """
        Find candidates to label.
        :param available_budget: The number of candidates to label
        :return: A list of the filenames of candidates to label
        """
        raise NotImplementedError


class LeastConfidenceActiveLearning(ActiveLearningModule):
    """
    An active learning module that chooses candidates to label based on confidence scores of examples.
    """

    def __init__(self, task):
        """
        Create a new LeastConfidenceActiveLearning module.
        :param task: The current task
        """
        super().__init__(task)

    def find_candidates(self, available_budget):
        """
        Find candidates to label based on confidence.
        :param available_budget: The number of candidates to label
        :return: A list of the filenames of candidates to label
        """
        raise NotImplementedError


class RandomActiveLearning(ActiveLearningModule):
    """
    An active learning module that randomly chooses candidates to label.
    """

    def __init__(self, task):
        """
        Create a new RandomActiveLearning module.
        :param task: The current task
        """
        super().__init__(task)
        self.labeled = set()  # List of candidates already labeled

    def find_candidates(self, available_budget):
        """
        Randomly find candidates to label.
        :param available_budget: The number of candidates to label
        :return: A list of the filenames of candidates to label
        """
        image_dir = self.task.unlabeled_image_path
        unlabeled_images = [f.name for f in Path(image_dir).iterdir() if f.is_file() and f.name not in self.labeled]
        to_request = sample(unlabeled_images, available_budget)
        self.labeled.update(to_request)
        return to_request
