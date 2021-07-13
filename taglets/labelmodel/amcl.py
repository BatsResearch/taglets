import numpy as np
import pandas as pd
import pickle
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
 
 
def get_data(num):
    '''
    Function to get the data from the DARPA task
    '''

    data = pickle.load(open("./saved_vote_matrices/cifar-1chkpnt.pkl", "rb"))

    data_dict = data["Base %d" % (num)]
    df = pd.read_feather("./cifar-labels_train.feather")

    print("Running Base %d" % (num))

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

   
def main(num_unlab, num_classes):
    '''
    Dylan's test script for evaluating AMCL (w/ convex combination of labelers + Briar score)

    Currently running on last year's DARPA eval - need to copy data to replicate

    You can change the amount of labeled data and unlabeled data by changing num_labeled_data and end_ind params.

    '''

    labelmodel = AMCLWeightedVote(num_classes)

    # loading last year's DARPA eval data for testing [MultiTaskModule, TransferModule, FineTuneModule, ZSLKGModule]
    l_names, l_labels, ul_names, ul_votes, id_class_dict = get_data(0)
    # test_names, test_votes, test_labels = get_test_data(1, base=True)

    ul_labels = [id_class_dict[x] for x in ul_names]
    
    num_labeled_data = len(l_names)

    # cutting off how much data we use
    num_labeled_data = 100
    end_ind = 100 + num_unlab
    
    ul_labels, ul_votes, ul_names = np.asarray(ul_labels), np.asarray(ul_votes), np.asarray(ul_names)
    indices = np.arange(len(ul_labels))
    np.random.shuffle(indices)

    # using the same amount of labeled data from unlabeled data since we don't have votes on original labeled data 
    l_labels = ul_labels[indices[:num_labeled_data]]
    l_votes = ul_votes[indices[:num_labeled_data]]
    l_names = ul_names[indices[:num_labeled_data]]


    ul_labels = ul_labels[indices[num_labeled_data:end_ind]]
    ul_votes = ul_votes[indices[num_labeled_data:end_ind]]
    ul_names = ul_names[indices[num_labeled_data:end_ind]]

    num_unlab = len(ul_names)

    print(np.shape(ul_votes))
    
    # restrict num classes
    ul_votes = np.minimum(ul_votes, num_classes - 1)
    l_votes = np.minimum(l_votes, num_classes - 1)

    cifar_classes = ['couch', 'otter', 'crab', 'boy', 'aquarium_fish', 'chimpanzee', 'telephone', 'cup', 'sweet_pepper',
                     'poppy', 'man', 'mountain', 'house', 'road', 'sunflower', 'sea', 'crocodile', 'rose',
                     'willow_tree', 'flatfish', 'possum', 'tractor', 'chair', 'bridge', 'wolf', 'elephant', 'fox',
                     'keyboard', 'beaver', 'tiger', 'baby', 'plate', 'rocket', 'turtle', 'streetcar', 'woman',
                     'caterpillar', 'forest', 'mouse', 'cattle', 'tulip', 'camel', 'pear', 'bicycle', 'lion', 'cloud',
                     'shrew', 'squirrel', 'porcupine', 'castle', 'clock', 'lizard', 'dolphin', 'orchid', 'television',
                     'snake', 'skyscraper', 'bee', 'trout', 'beetle', 'worm', 'lamp', 'tank', 'maple_tree', 'whale',
                     'kangaroo', 'orange', 'table', 'bed', 'lobster', 'palm_tree', 'raccoon', 'pickup_truck',
                     'pine_tree', 'butterfly', 'lawn_mower', 'dinosaur', 'ray', 'can', 'mushroom', 'motorcycle',
                     'apple', 'seal', 'hamster', 'shark', 'skunk', 'plain', 'bowl', 'train', 'bear', 'leopard', 'girl',
                     'cockroach', 'spider', 'rabbit', 'bottle', 'snail', 'bus', 'oak_tree', 'wardrobe']

    class_to_ind = {x: i for i, x in enumerate(cifar_classes)}
    
    l_labels = [class_to_ind[x] for x in l_labels]
    ul_labels = [class_to_ind[x] for x in ul_labels]
    l_labels = np.minimum(l_labels, num_classes - 1)
    ul_labels = np.minimum(ul_labels, num_classes - 1)
    # print("Num Labeled: %d" % (num_labeled_data))
    # print("Num Unlabeled: %d" % (num_unlab))
    # print("L Votes", np.shape(l_votes))
    # print("UL Votes", np.shape(ul_votes))

    print('Training...', flush=True)
    labelmodel.train(l_votes, l_labels, ul_votes)
    preds = labelmodel.get_weak_labels(ul_votes)
    predictions = np.asarray([np.argmax(pred) for pred in preds])
    print("Acc %f" % (np.mean(predictions == ul_labels)))
    print(np.shape(preds), np.shape(ul_labels))

if __name__ == "__main__":
    import time
    for num_unlab in [1000]:
        for num_classes in np.arange(10, 101, 10):
            for chkpnt in [0]:
                st = time.time()
                print(f'-------------- {num_unlab} unlabeled images and {num_classes} classes at chkpont {chkpnt}---------------', flush=True)
                main(num_unlab, num_classes)
                print(f'-------------- Elapsed: {(time.time() - st) / 60.0} mins -----------------------------', flush=True)
