class ScadsNode:
    """
    A class to represent a node in the SCADS
    """
    def __init__(self, concept, wordId):
        self.concept = concept
        self.wordId = wordId
        self.datasets = []  # list of our datasets containing this concept

    def get_datasets(self):
        """
        Get list of datasets.
        :return: List of our datasets
        """
        raise NotImplementedError()

    def get_images(self):
        """
        Jeff: Is this returning the list of paths to images, or the actual numpy arrays?

        Get all images for this concept.
        :return: List of images for this concept
        """
        raise NotImplementedError()

    def get_neighbors(self):
        """
        Get the neighbors of this concept with the type of relationship.
        :return: List of ScadsEdges
        """
        raise NotImplementedError()
