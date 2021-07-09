from .label_model import LabelModel

from labelmodels import PartialLabelModel


class NaiveBayes(LabelModel):
    def get_weak_labels(self, vote_matrix, *args):
        """
        Combine votes using a Naive Bayes label model.

        :param vote_matrix: m x n matrix in {0, ..., k-1}, where m is the number
                            of examples, n is the number of taglets and k is the
                            number of classes
        :param weights: list-like of k elements in [0,1]
        :return: m x k matrix, where each row is a distribution over the k
                 possible labels
        """
        # Increment votes to match convention in labelmodels package
        vote_matrix = vote_matrix.copy() + 1

        # Creates label_partition for configuring as a regular label model
        # i.e., no partial labels
        label_partition = {}
        for j in range(vote_matrix.shape[1]):
            label_partition[j] = [[i+1] for i in range(j)]

        # Initializes label model
        lm = PartialLabelModel(
            num_classes=self.num_classes, label_partition=label_partition
        )

        # Trains label model
        lm.estimate_label_model(vote_matrix)

        return lm.get_label_distribution(vote_matrix)