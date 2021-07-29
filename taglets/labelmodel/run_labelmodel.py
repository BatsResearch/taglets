import argparse
import numpy as np
import pandas as pd
import pickle
import time
import os
import torchvision.transforms as transforms

from .amcl import AMCLWeightedVote, AMCLLogReg
from .naive_bayes import NaiveBayes
from .weighted import WeightedVote, UnweightedVote
from ..data import CustomImageDataset


def get_data(num):
    '''
    Function to get the data from the DARPA task
    '''
    
    data = pickle.load(open("./saved_vote_matrices/soft_cifar_votes", "rb"))
    
    data_dict = data["Base %d" % (num)]
    df = pd.read_feather("./cifar-labels_train.feather")
    
    print("Running Base %d" % (num))
    
    l_names = data_dict["labeled_images_names"]
    l_votes = data_dict["labeled_images_votes"]
    l_labels = data_dict["labeled_images_labels"]
    ul_names = data_dict["unlabeled_images_names"]
    ul_votes = data_dict["unlabeled_images_votes"]
    id_class_dict = pd.Series(df["class"].values, index=df.id.values).to_dict()

    l_labels, l_votes, l_names = np.asarray(l_labels), np.asarray(l_votes), np.asarray(l_names)
    ul_votes, ul_names = np.asarray(ul_votes), np.asarray(ul_names)
    indices = np.arange(len(ul_names))
    np.random.shuffle(indices)
    num_labeled_data = 2000
    
    l_votes = ul_votes[:, indices[:num_labeled_data]]
    l_names = ul_names[indices[:num_labeled_data]]
    
    ul_votes = ul_votes[:, indices[num_labeled_data:]]
    ul_names = ul_names[indices[num_labeled_data:]]
    
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
    if labelmodel_type == 'amcl-lr':
        labelmodel = AMCLLogReg(num_classes)
    elif labelmodel_type == 'amcl-cc':
        labelmodel = AMCLWeightedVote(num_classes)
    elif labelmodel_type == 'naive_bayes':
        labelmodel = NaiveBayes(num_classes)
    elif labelmodel_type == 'weighted':
        labelmodel = WeightedVote(num_classes)
    else:
        labelmodel = UnweightedVote(num_classes)
    
    # loading last year's DARPA eval data for testing [MultiTaskModule, TransferModule, FineTuneModule, ZSLKGModule]
    l_names, l_votes, l_labels, ul_names, ul_votes, id_class_dict = get_data(chkpnt)
    # test_names, test_votes, test_labels = get_test_data(1, base=True)

    
    ul_image_paths = [os.path.join('/users/wpiriyak/data/bats/datasets/lwll/external/cifar100/cifar100_full/train',
                                image_name) for image_name in ul_names]
    
    l_labels = [id_class_dict[x] for x in l_names]
    ul_labels = [id_class_dict[x] for x in ul_names]
    
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
    
    # cutting off how much data we use
    
    ul_labels, ul_votes, ul_names, ul_image_paths = np.asarray(ul_labels), np.asarray(ul_votes), np.asarray(ul_names), \
                                                    np.asarray(ul_image_paths)
    indices = np.arange(len(ul_labels))
    np.random.shuffle(indices)
    
    if num_unlab == -1:
        sampled_ul_votes = ul_votes
        sampled_ul_image_paths = ul_image_paths
    else:
        sampled_ul_votes = ul_votes[:, indices[:num_unlab]]
        sampled_ul_image_paths = ul_image_paths[indices[:num_unlab]]

    data_mean = [0.485, 0.456, 0.406]
    data_std = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=data_mean, std=data_std)
    ])
    sampled_ul_dataset = CustomImageDataset(sampled_ul_image_paths, transform=transform)
    ul_dataset = CustomImageDataset(ul_image_paths, transform=transform)
    
    print(np.shape(ul_votes))
    
    # restrict num classes
    sampled_ul_votes = np.minimum(sampled_ul_votes, num_classes - 1)
    l_votes = np.minimum(l_votes, num_classes - 1)
    l_labels = np.minimum(l_labels, num_classes - 1)
    
    for i in range(l_votes.shape[0]):
        print(f'Labeled acc for module {i}: {np.mean(np.argmax(l_votes[i], 1) == l_labels)}')
        
    print('Training...', flush=True)
    
    if labelmodel_type == 'amcl-lr':
        labelmodel.train(l_votes, l_labels, sampled_ul_votes, sampled_ul_dataset)
        print(f'Thetas: {labelmodel.theta}')
    elif labelmodel_type == 'amcl-cc':
        labelmodel.train(l_votes, l_labels, sampled_ul_votes)
        print(f'Thetas: {labelmodel.theta}')

    if labelmodel_type == 'amcl-lr':
        preds = labelmodel.get_weak_labels(ul_dataset)
    elif labelmodel_type == 'weighted':
        preds = labelmodel.get_weak_labels(ul_votes,
                                           [np.mean(np.argmax(l_votes[i], 1) == l_labels) for i in range(l_votes.shape[0])])
    else:
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