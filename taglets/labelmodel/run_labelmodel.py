import argparse
import numpy as np
import pandas as pd
import pickle
import time

from .amcl import AMCLWeightedVote
from .naive_bayes import NaiveBayes

def get_data(num):
    '''
    Function to get the data from the DARPA task
    '''
    
    data = pickle.load(open("./saved_vote_matrices/cifar-1chkpnt.pkl", "rb"))
    
    data_dict = data["Base %d" % (num)]
    df = pd.read_feather("./cifar-labels_train.feather")
    
    print("Running Base %d" % (num))
    
    l_names = data_dict["labeled_images_names"]
    l_votes = data_dict["labeled_images_votes"]
    l_labels = data_dict["labeled_images_labels"]
    ul_names = data_dict["unlabeled_images_names"]
    ul_votes = data_dict["unlabeled_images_votes"]
    id_class_dict = pd.Series(df["class"].values, index=df.id.values).to_dict()
    
    return l_names, l_votes, l_labels, ul_names, ul_votes, id_class_dict


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


def run_one_checkpoint(num_unlab, num_classes, chkpnt, labelmodel_type='amcl'):
    '''
    Dylan's test script for evaluating AMCL (w/ convex combination of labelers + Briar score)

    Currently running on last year's DARPA eval - need to copy data to replicate

    You can change the amount of labeled data and unlabeled data by changing num_labeled_data and end_ind params.

    '''
    if labelmodel_type == 'amcl':
        labelmodel = AMCLWeightedVote(num_classes)
    else:
        labelmodel = NaiveBayes(num_classes)
    
    # loading last year's DARPA eval data for testing [MultiTaskModule, TransferModule, FineTuneModule, ZSLKGModule]
    l_votes, l_labels, ul_votes, ul_labels = get_data(chkpnt)
    
    # cutting off how much data we use
    
    ul_labels, ul_votes = np.asarray(ul_labels), np.asarray(ul_votes)
    indices = np.arange(len(ul_labels))
    np.random.shuffle(indices)
    
    if num_unlab == -1:
        sampled_ul_votes = ul_votes
    else:
        sampled_ul_votes = ul_votes[indices[:num_unlab]]
    
    print(np.shape(ul_votes))
    
    # restrict num classes
    sampled_ul_votes = np.minimum(sampled_ul_votes, num_classes - 1)
    l_votes = np.minimum(l_votes, num_classes - 1)
    l_labels = np.minimum(l_labels, num_classes - 1)
    
    print('Training...', flush=True)
    if labelmodel_type == 'amcl':
        labelmodel.train(l_votes, l_labels, sampled_ul_votes)
    preds = labelmodel.get_weak_labels(ul_votes)
    predictions = np.asarray([np.argmax(pred) for pred in preds])
    print("Acc %f" % (np.mean(predictions == ul_labels)), flush=True)
    print(np.shape(preds), np.shape(ul_labels))


def main():
    parser = argparse.ArgumentParser(description="Run JPL task")
    parser.add_argument("--labelmodel",
                        type=str,
                        default="amcl")
    args = parser.parse_args()
    
    for num_unlab in [1000]:
        for num_classes in [100]:
            for chkpnt in [0, 1, 2, 3]:
                st = time.time()
                print(
                    f'-------------- {num_unlab} unlabeled images and {num_classes} classes at chkpont {chkpnt}---------------',
                    flush=True)
                run_one_checkpoint(num_unlab, num_classes, chkpnt, args.labelmodel)
                print(f'-------------- Elapsed: {(time.time() - st) / 60.0} mins -----------------------------',
                      flush=True)
                

if __name__ == '__main__':
    main()