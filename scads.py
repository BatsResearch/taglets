import numpy as np


class Scads:
    """
    Scads class
    Structured Collections of Annotated Data Sets (SCADS)
    """
    def __init__(self):
        self.concept = ''
        self.datasets = []  # list of our datasets

    def get_datasets(self):
        """
        Update list of datasets.
        :return:
        """

    def get_images(self, concept):
        """
        Jeff: Is the concept (here and in get_neighbors) just a wordId?
        Jeff: Also, is this returning the list of paths to images, or the actual numpy arrays?

        Get all images for a concept.
        :param concept:
        :return:
        """

    def get_neighbors(self, concept):
        """
        Get the neighbors of a concept with the type of relationship.
        :param concept:
        :return: List of (Concept, Relationship) tuples
        """
