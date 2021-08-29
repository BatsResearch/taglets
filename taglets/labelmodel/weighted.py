from .label_model import LabelModel

import numpy as np


class UnweightedVote(LabelModel):
    def get_weak_labels(self, vote_matrix, *args):
        """
        Combine votes by giving each taglet equal weight

        :param vote_matrix: m x n matrix in {0, ..., k-1}, where m is the number
                            of examples, n is the number of taglets and k is the
                            number of classes
        :return: m x k matrix, where each row is a distribution over the k
                 possible labels
        """
        weights = [1.0] * vote_matrix.shape[0]
        return self._get_weighted_dist(vote_matrix, weights)

    def _get_weighted_dist(self, vote_matrix, weights):
        weak_labels = []
        for j in range(vote_matrix.shape[1]):
            weak_label = np.zeros((self.num_classes,))
            for i in range(vote_matrix.shape[0]):
                weak_label += weights[i] * vote_matrix[i][j]
            weak_labels.append(weak_label / sum(weights))
        return weak_labels


class WeightedVote(UnweightedVote):
    def get_weak_labels(self, vote_matrix, weights, *args):
        """
        Combine votes giving each taglet a specific weight in [0, 1].

        :param vote_matrix: m x n matrix in {0, ..., k-1}, where m is the number
                            of examples, n is the number of taglets and k is the
                            number of classes
        :param weights: list-like of k elements in [0,1]
        :return: m x k matrix, where each row is a distribution over the k
                 possible labels
        """
        return self._get_weighted_dist(vote_matrix, weights)
