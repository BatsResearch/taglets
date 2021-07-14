import numpy as np
from .amcl_helper import compute_constraints_with_loss, Brier_loss_linear, linear_combination_labeler
from .amcl_helper import compute_constraints_with_loss2, Brier_Score_AMCL, linear_combination_labeler
from .amcl_helper import projectToSimplex, projectToBall, projectCC
from .amcl_helper import subGradientMethod, subGradientMethod2
from .weighted import UnweightedVote


class AMCLWeightedVote(UnweightedVote):
    """
    Class representing a weighted vote of supervision sources trained through AMCL
    """
    def __init__(self, num_classes):
        super().__init__(num_classes)
        self.theta = None
    
    def train(self, labeled_vote_matrix, labels, unlabeled_vote_matrix):
        '''
        Train a convex combination of weak supervision sources through AMCL

        Args:
        labeled_vote_matrix - outputs on labeled data (# wls, # l data, # classes)
        labels - true labels on labeled data
        unlabeled_vote_matrix - outputs on unlabeled data (# wls, # ul data, # classes)
        '''
        # pre process vote matrix
        # convert votes to one hot
        labeled_vote_matrix = np.eye(self.num_classes)[labeled_vote_matrix]
        unlabeled_vote_matrix = np.eye(self.num_classes)[unlabeled_vote_matrix]

        labeled_vote_matrix = np.transpose(labeled_vote_matrix, (1, 0, 2))
        unlabeled_vote_matrix = np.transpose(unlabeled_vote_matrix, (1, 0, 2))
        
        labels = np.eye(self.num_classes)[labels]

        # hyperparameters
        N = 4 # of wls
        eps = 0.3
        L = 2 * np.sqrt(N + 1)
        squared_diam = 2
        T = 150
        h = eps/(L*L)

        # assuming structure of vote matrix is (# wls, # data, # classes)
        self.num_wls, num_unlab, _ = np.shape(unlabeled_vote_matrix)
        self.theta = np.ones(self.num_wls) * (1 / self.num_wls)

        # generate constraints
        constraint_matrix, constraint_vector, constraint_sign = compute_constraints_with_loss(Brier_loss_linear, 
                                                                                                 unlabeled_vote_matrix, 
                                                                                                 labeled_vote_matrix, 
                                                                                                 labels)

        self.theta = subGradientMethod(unlabeled_vote_matrix, constraint_matrix, constraint_vector, 
                                          constraint_sign, Brier_loss_linear, linear_combination_labeler, 
                                          projectToSimplex, self.theta, 
                                          T, h, N, num_unlab, self.num_classes)

        # cvxpy implementation
        # Y, constraints = compute_constraints_with_loss2(Brier_loss_linear, Brier_Score_AMCL, unlabeled_vote_matrix, 
                                                        # labeled_vote_matrix, labels)

        # self.theta = subGradientMethod2(unlabeled_vote_matrix, Y, constraints, Brier_loss_linear, linear_combination_labeler, 
                                        # projectToSimplex, self.theta, T, h, N, num_unlab, self.num_classes)

    def get_weak_labels(self, vote_matrix, *args):
        return self._get_weighted_dist(vote_matrix, self.theta)
