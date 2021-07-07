import numpy as np
import pandas as pd
import pickle
from amcl_helper import compute_constraints_with_loss, Brier_loss_linear, linear_combination_labeler
from amcl_helper import compute_constraints_with_loss2, Brier_Score_AMCL, linear_combination_labeler
from amcl_helper import projectToSimplex, projectToBall, projectCC 
from amcl_helper import subGradientMethod, subGradientMethod2
from label_model import LabelModel


class AMCLWeightedVote(LabelModel):
    """
    Class representing a weighted vote of supervision sources trained through AMCL
    """

    def __init__(self, num_classes):
        self.theta = None
        self.num_classes = num_classes
    
    def train(self, labeled_vote_matrix, labels, unlabeled_vote_matrix):
        '''
        Train a convex combination of weak supervision sources through AMCL

        Args:
        labeled_vote_matrix - outputs on labeled data (# wls, # l data, # classes)
        labels - true labels on labeled data
        unlabeled_vote_matrix - outputs on unlabeled data (# wls, # ul data, # classes)
        '''

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

    def get_weak_labels(self, vote_matrix):
        
        # This should aggregate the votes from various taglets and output labels for the unlabeled data

        num_unlab = np.shape(vote_matrix)[1]
        preds = np.zeros((num_unlab, self.num_classes))

        for i in range(self.num_wls):
            preds += self.theta[i] * vote_matrix[i, :, :]

        return np.argmax(preds, axis=1)
 
def get_data(num, base=True):
    '''
    Function to get the data from the DARPA task
    '''

    data = pickle.load(open("./ta2.pkl", "rb"))  

    if base:
        data_dict = data["Base %d" % (num)]
        df = pd.read_feather("./domain_net-clipart_labels_train.feather")
    
        print("Running Base %d" % (num))
    
    else:
        data_dict = data["Adapt %d" % (num)]
        df = pd.read_feather("./domain_net-sketch_labels_train.feather")
    
        print("Running Adapt %d" % (num))

    l_names = data_dict["labeled_images_names"]
    l_labels = data_dict["labeled_images_labels"]
    ul_names = data_dict["unlabeled_images_names"]
    ul_votes = data_dict["unlabeled_images_votes"]
    id_class_dict = pd.Series(df["class"].values, index=df.id.values).to_dict()

    return l_names, l_labels, ul_names, ul_votes, id_class_dict

def get_test_data(num, base=True):

    data = pickle.load(open("./ta2_test_votes_full.pkl", "rb"))
    if base:
        data_dict = data["Base %d" % (num)]
        df = pd.read_feather("./domain_net-clipart_labels_test.feather")
    else:
        data_dict = data["Adapt %d" % (num)]
        df = pd.read_feather("./domain_net-sketch_labels_test.feather")

    test_names = data_dict["unlabeled_images_names"]
    test_votes = data_dict["unlabeled_images_votes"]
    id_class_dict = pd.Series(df["class"].values, index=df.id.values).to_dict()

    test_labels = [id_class_dict[x] for x in test_names]
    return test_names, test_votes, test_labels

   
def main():
    '''
    Dylan's test script for evaluating AMCL (w/ convex combination of labelers + Briar score)

    Currently running on last year's DARPA eval - need to copy data to replicate

    You can change the amount of labeled data and unlabeled data by changing num_labeled_data and end_ind params.

    '''

    num_classes = 345
    labelmodel = AMCLWeightedVote(num_classes)
    base = True

    # loading last year's DARPA eval data for testing [MultiTaskModule, TransferModule, FineTuneModule, ZSLKGModule]
    l_names, l_labels, ul_names, ul_votes, id_class_dict = get_data(1, base=True)
    test_names, test_votes, test_labels = get_test_data(1, base=True)

    ul_labels = [id_class_dict[x] for x in ul_names]
    
    num_labeled_data = len(l_names)

    # cutting off how much data we use
    num_labeled_data = 100
    end_ind = 150

    # using the same amount of labeled data from unlabeled data since we don't have votes on original labeled data 
    l_labels = ul_labels[:num_labeled_data]
    l_votes = ul_votes[:num_labeled_data]
    l_names = ul_names[:num_labeled_data]


    ul_labels = ul_labels[num_labeled_data:end_ind]
    ul_votes = ul_votes[num_labeled_data:end_ind]
    ul_names = ul_names[num_labeled_data:end_ind]

    num_unlab = len(ul_names)

    # converting votes to one-hots
    l_votes = np.eye(num_classes)[l_votes]
    ul_votes = np.eye(num_classes)[ul_votes]

    l_votes = np.transpose(l_votes, (1, 0, 2))
    ul_votes = np.transpose(ul_votes, (1, 0, 2))

    print(np.shape(ul_votes)    )

    clipart_classes = pickle.load(open("./domain_net-clipart_classes.pkl", "rb"))
    sketch_classes = pickle.load(open("./domain_net-sketch_classes.pkl", "rb"))

    base_class_to_ind = {x: i for i, x in enumerate(clipart_classes)}
    adapt_class_to_ind =  {x: i for i, x in enumerate(sketch_classes)}

    if base == 1:
        l_labels = [base_class_to_ind[x] for x in l_labels]
        ul_labels = [base_class_to_ind[x] for x in ul_labels]
    else:
        l_labels = [adapt_class_to_ind[x] for x in l_labels]
        ul_labels = [adapt_class_to_ind[x] for x in ul_labels]


    l_labels = np.eye(num_classes)[l_labels]
    ul_labels = np.eye(num_classes)[ul_labels]

    # print("Num Labeled: %d" % (num_labeled_data))
    # print("Num Unlabeled: %d" % (num_unlab))
    # print("L Votes", np.shape(l_votes))
    # print("UL Votes", np.shape(ul_votes))

    labelmodel.train(l_votes, l_labels, ul_votes)
    preds = labelmodel.get_weak_labels(ul_votes)
    print("Acc %f" % (np.mean(preds == np.argmax(ul_labels, 1))))
    print(np.shape(preds), np.shape(ul_labels))

if __name__ == "__main__":
    main()
