from random import sample


class ActiveLearningModule:
    """
    The base class for an active learning module. Used to find examples to label.
    """

    def find_candidates(self, available_budget, unlabeled_image_names):
        """
        Find candidates to label.
        :param available_budget: The number of candidates to label
        :param unlabeled_image_names: The name of unlabeled images
        :return: A list of the filenames of candidates to label
        """
        raise NotImplementedError


class LeastConfidenceActiveLearning(ActiveLearningModule):
    """
    An active learning module that chooses candidates to label based on confidence scores of examples.
    """

    def __init__(self):
        """
        Create a new LeastConfidenceActiveLearning module.
        :param task: The current task
        """
        super().__init__()
        self.next_candidates = []

    def set_candidates(self, candidates):
        self.next_candidates = candidates

    def find_candidates(self, available_budget, unlabeled_image_names):
        """
        Find candidates to label based on confidence.
        :param available_budget: The number of candidates to label
        :param unlabeled_image_names: The name of unlabeled images
        :return: A list of the filenames of candidates to label
        """
        return list(map(unlabeled_image_names.__getitem__, self.next_candidates[:available_budget]))


class RandomActiveLearning(ActiveLearningModule):
    """
    An active learning module that randomly chooses candidates to label.
    """

    def find_candidates(self, available_budget, unlabeled_image_names):
        """
        Randomly find candidates to label.
        :param available_budget: The number of candidates to label
        :param unlabeled_image_names: The name of unlabeled images
        :return: A list of the filenames of candidates to label
        """
        return sample(unlabeled_image_names, min(len(unlabeled_image_names), available_budget))
