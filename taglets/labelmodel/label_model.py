class LabelModel:
    """
    A model for combining votes from taglets into probabilistic training labels,
    perhaps using additional side information.
    """

    def __init__(self, num_classes):
        """
        Instantiate a label model.

        :param num_classes:
        """
        self.num_classes = num_classes

    def get_weak_labels(self, vote_matrix, *args):
        """
        Main method for combining taglets' outputs into weak labels. Subclasses
        may add additional arguments for required side information.

        :param vote_matrix: m x n matrix in {0, ..., k-1}, where m is the number
                            of examples, n is the number of taglets and k is the
                            number of classes
        :return: m x k matrix, where each row is a distribution over the k
                 possible labels
        """
        raise NotImplementedError
