import os
import torch
import random
import pickle
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
import torchvision.transforms as transforms

import clip
from PIL import Image

def dataset_object(dataset_name):

    if dataset_name == 'aPY':
        from .data import aPY as DataObject
    elif dataset_name == 'Animals_with_Attributes2':
        from .data import AwA2 as DataObject
    elif dataset_name == 'EuroSAT':
        from .data import EuroSAT as DataObject

    return DataObject

log = logging.getLogger(__name__)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class Config(object):
    def __init__(self, config):
        for k, v in config.items():
            setattr(self, k, v)   

def get_class_names(dataset, dataset_dir):
    """ Returns the lists of the names of all classes, seen classes,
    and unseen classes.
    
    :param dataset: name of the dataset in use
    :param dataset_dir: path to get dataset dir (on CCV)
    """
    if dataset == 'aPY':
        path = f"{dataset_dir}/{dataset}/proposed_split"
        seen_classes = []
        unseen_classes = []
        with open(f"{path}/trainvalclasses.txt", 'r') as f:
            for l in f:
                seen_classes.append(l.strip())
        
        with open(f"{path}/testclasses.txt", 'r') as f:
            for l in f:
                unseen_classes.append(l.strip())

        # Adjust class names
        correction_dict = {'diningtable': 'dining table',
                           'tvmonitor': 'tv monitor',
                           'jetski': 'jet ski',
                           'pottedplant': 'potted plant'}
        for c in seen_classes:
            if c in correction_dict:
                seen_classes[seen_classes.index(c)] = correction_dict[c]
        for c in unseen_classes:
            if c in correction_dict:
                unseen_classes[unseen_classes.index(c)] = correction_dict[c]

        classes = seen_classes + unseen_classes

    elif dataset == 'Animals_with_Attributes2':
        path = f"{dataset_dir}/{dataset}"

        seen_classes = []
        unseen_classes = []
        df = pd.read_csv(f"{path}/trainvalclasses.txt")
        with open(f"{path}/trainvalclasses.txt", 'r') as f:
            for l in f:
                seen_classes.append(l.strip())
        
        with open(f"{path}/testclasses.txt", 'r') as f:
            for l in f:
                unseen_classes.append(l.strip())

        # Adjust class names
        correction_dict = {'grizzly+bear': 'grizzly bear',
                           'killer+whale': 'killer whale',
                           'persian+cat': 'persian cat',
                           'german+shepherd': 'german shepherd',
                           'blue+whale': 'blue whale',
                           'siamese+cat': 'siamese cat',
                           'spider+monkey': 'spider monkey',
                           'humpback+whale': 'humpback whale',
                           'giant+panda': 'giant panda',
                            'polar+bear': 'polar bear'}
        
        for c in seen_classes:
            if c in correction_dict:
                seen_classes[seen_classes.index(c)] = correction_dict[c]
        for c in unseen_classes:
            if c in correction_dict:
                unseen_classes[unseen_classes.index(c)] = correction_dict[c]

        classes = seen_classes + unseen_classes

    elif dataset == 'EuroSAT':
        path = f"{dataset_dir}/{dataset}"

        classes = []
        with open(f"{path}/class_names.txt", 'r') as f:
            for l in f:
                classes.append(l.strip())
        
        np.random.seed(500)
        seen_indices = np.random.choice(range(len(classes)),
                                size=int(len(classes)*0.62),
                                replace=False)
        unseen_indices = list(set(range(len(classes))).difference(set(seen_indices)))

        seen_classes = list(np.array(classes)[seen_indices])
        unseen_classes = list(np.array(classes)[unseen_indices])

    return classes, seen_classes, unseen_classes

def get_labeled_and_unlabeled_data(dataset, data_folder, 
                                   seen_classes, unseen_classes):
    """ This function returns the list of
    - labeled_data: each item is (image name, class name)
    - unlabeled_data: each item is (image name, class name)
    - test_data: each item is (image name, class name)
    
    :param dataset: dataset name   
    :param data_folder: path to folder of images
    :param seen_classes: list of seen classes' names
    :param unseen_classes: list of unseen classes' names
    """
    if dataset == 'aPY':
        image_data = pd.read_csv(f"{data_folder}/image_data.csv", sep=',')
        
        list_images = []
        for i, row in image_data.iterrows():
            if row['image_path'] == 'yahoo_test_images/bag_227.jpg' or \
                row['image_path'] == 'yahoo_test_images/mug_308.jpg':
                list_images.append(f'broken')
            else:
                list_images.append(f'{i}.jpg')

        image_data['file_names'] = list_images
        correction_dict = {'diningtable': 'dining table',
                           'tvmonitor': 'tv monitor',
                           'jetski': 'jet ski',
                           'pottedplant': 'potted plant'}
        image_data['label'] = image_data['label'].apply(lambda x: correction_dict[x] if x in correction_dict else x)
        image_data['seen'] = image_data['label'].apply(lambda x: 1 if x in seen_classes else 0)
        
        labeled_files = list(image_data[(image_data['seen'] == 1) & (image_data['file_names'] != 'broken')]['file_names'])
        labels_files = list(image_data[(image_data['seen'] == 1) & (image_data['file_names'] != 'broken')]['label'])
        log.info(f"NUMBER OF UNIQUE SEEN CLASSES: {len(set(labels_files))}")


        unlabeled_lab_files = list(image_data[(image_data['seen'] == 0) & (image_data['file_names'] != 'broken')]['file_names'])
        unlabeled_labs = list(image_data[(image_data['seen'] == 0) & (image_data['file_names'] != 'broken')]['label'])
        log.info(f"NUMBER OF UNIQUE UNSEEN CLASSES: {len(set(unlabeled_labs))}")

    elif dataset == 'Animals_with_Attributes2':
        labeled_files = []
        labels_files = []
        for c in seen_classes:
            files = os.listdir(f"{data_folder}/JPEGImages/{c.replace(' ', '+')}")
            labeled_files += files
            labels_files += [c]*len(files)

        unlabeled_lab_files = []
        unlabeled_labs = []
        for c in unseen_classes:
            files = os.listdir(f"{data_folder}/JPEGImages/{c.replace(' ', '+')}")
            unlabeled_lab_files += files
            unlabeled_labs += [c]*len(files)

    elif dataset == 'EuroSAT':

        correction_dict = {'Annual Crop Land': 'AnnualCrop',
                           'Herbaceous Vegetation Land': 'HerbaceousVegetation',
                           'Highway or Road': 'Highway',
                           'Industrial Buildings': 'Industrial',
                           'Pasture Land': 'Pasture',
                           'Permanent Crop Land': 'PermanentCrop',
                           'Residential Buildings': 'Residential',
                           'Sea or Lake': 'SeaLake',
                           'River': 'River',
                           'Forest', 'Forest'}
        
        labeled_files = []
        labels_files = []
        for c in seen_classes:
            files = os.listdir(f"{data_folder}/{correction_dict[c]}")
            labeled_files += files
            labels_files += [c]*len(files)

        unlabeled_lab_files = []
        unlabeled_labs = []
        for c in unseen_classes:
            files = os.listdir(f"{data_folder}/{correction_dict[c]}")
            unlabeled_lab_files += files
            unlabeled_labs += [c]*len(files)
        
    # Split labeled and unlabeled data into test
    train_labeled_files, train_labeles, test_seen_files, test_seen_labs = split_data(0.8, labeled_files, labels_files)
    labeled_data = list(zip(train_labeled_files, train_labeles))

    train_unlabeled_files, train_un_labeles, test_unseen_files, test_unseen_labs = split_data(0.8, unlabeled_lab_files, unlabeled_labs)
    unlabeled_data = list(zip(train_unlabeled_files, train_un_labeles))

    test_seen = list(zip(test_seen_files, test_seen_labs))
    test_unseen = list(zip(test_unseen_files, test_unseen_labs))
    
    test_data = test_seen + test_unseen

    return labeled_data, unlabeled_data, test_data


def split_data(ratio, files, labels):
    np.random.seed(500)
    train_indices = np.random.choice(range(len(files)),
                                size=int(len(files)*ratio),
                                replace=False)
    val_indices = list(set(range(len(files))).difference(set(train_indices)))

    train_labeled_files = np.array(files)[train_indices]
    train_labeles = np.array(labels)[train_indices]

    val_labeled_files = np.array(files)[val_indices]
    val_labeles = np.array(labels)[val_indices]

    return train_labeled_files, train_labeles, val_labeled_files, val_labeles


def parse_transform(transform, image_size=224):
    if transform == 'RandomResizedCrop':
        return transforms.RandomResizedCrop(image_size)
    elif transform == 'RandomColorJitter':
        return transforms.RandomApply([transforms.ColorJitter(0.6, 0.8, 0.2, 0.5)], p=1.0) # transforms.ColorJitter(brightness=.2, contrast=.1, saturation=.1, hue=.1),
    elif transform == 'RandomGrayscale':
        return transforms.RandomGrayscale(p=0.4)
    elif transform == 'RandomHorizontalFlip':
        return transforms.RandomHorizontalFlip(p=0.5)
    elif transform == 'RandomGaussianBlur':
        return transforms.RandomApply([transforms.GaussianBlur(kernel_size=(9,9))], p=1)
    elif transform == 'Resize':
        return transforms.Resize([image_size, image_size])
    elif transform == 'RandomAffine1':
        return transforms.RandomAffine(degrees=30, translate=(0.15, 0.15),
                                       shear=(-30, 30, -30, 30))
    elif transform == 'RandomAffine2':
        return transforms.RandomAffine(degrees=60, translate=(0.35, 0.35),
                                       shear=(-30, 30, -30, 30))

def composed_transform(augmentation, transform_base=None,
                       transform_strong=None, preprocess_transform=None, 
                       image_size=224):
    if augmentation == 'base':
        transform_list = transform_base
    elif augmentation == 'strong':
        transform_list = transform_strong
    elif augmentation == None:
        transform_list = []

    transform_fn = [parse_transform(t, image_size) for t in transform_list]
    if preprocess_transform is not None:
        transform = transforms.Compose(transform_fn + [preprocess_transform])
    else:
        transform = transforms.Compose(transform_fn)
    
    return transform

def prepare_data_ssl_loss(aug_1, aug_2):
        
        # Concatenate prompts, a positive pair is concat_prompts[i] 
        # and concat_prompts[i+batch_size]
        concat_prompts = torch.cat((aug_1, aug_2), dim=0)
        # Similarity labels
        labels = torch.arange(concat_prompts.shape[0] // 2) # creates [0, ..., batch size]
        labels = torch.cat((labels, labels)) # concat labels for pairing

        return concat_prompts, labels


def compute_pseudo_labels(k, template, dataset, 
                      classnames, transform, clip_model,
                      label_to_idx, device, filename):

    prompts = [f"{template}{' '.join(i.split('_'))}" \
                            for i in classnames]
    text = clip.tokenize(prompts).to(device)
    
    # to find the top k for each class, each class has it's own "leaderboard"
    top_k_leaderboard = {label_to_idx[classnames[i]] : [] 
                            for i in range(len(classnames))} #maps class idx -> (confidence, image_path) tuple
    
    log.info(f"Compute {k} pseudo-labeles")
    #log.info(f"{label_to_idx}")
    for i, image_path in enumerate(tqdm(dataset.filepaths)):
        img = Image.open(image_path).convert('RGB')
        img = transform(img).to(device)
        with torch.no_grad():
            logits_per_image, logits_per_text = clip_model(torch.unsqueeze(img, 0).to(device), text)
            probs = logits_per_image.softmax(dim=-1)
            idx_preds = torch.argmax(probs, dim=1)
            pred_id = idx_preds.item()
            pred = label_to_idx[classnames[idx_preds.item()]]
            #log.info(f"{classnames[idx_preds.item()]}")
            #log.info(f"{pred}")

        """if predicted class has empty leaderboard, or if the confidence is high
        enough for predicted class leaderboard, add the new example
        """
        prob_score = probs[0][pred_id]
        if len(top_k_leaderboard[pred]) < k:
                top_k_leaderboard[pred].append((prob_score, image_path))
        elif top_k_leaderboard[pred][-1][0] < prob_score: #if the confidence in predicted class "qualifies" for top-k
                # default sorting of tuples is by first element
                top_k_leaderboard[pred] = sorted(top_k_leaderboard[pred] + [(probs[0][pred_id], image_path)], reverse=True)[:k]
        else:
            #sort the other classes by confidence score
            order_of_classes = sorted([(probs[0][j], j) for j in range(len(classnames)) if j != pred_id], reverse=True)
            for score, index in order_of_classes:
                index_dict = label_to_idx[classnames[index]]
                #log.info(f"{classnames[index]}")
                #log.info(f"{index_dict}")
                if len(top_k_leaderboard[index_dict]) < k:
                    top_k_leaderboard[index_dict].append((probs[0][index], image_path))
                elif top_k_leaderboard[index_dict][-1][0] < probs[0][index]:
                    #default sorting of tuples is by first element
                    top_k_leaderboard[index_dict] = sorted(top_k_leaderboard[index_dict] + [((probs[0][index], image_path))], reverse=True)[:k]
    
    old_dataset = dataset
    new_imgs = []
    new_labels = []
    #loop through, and rebuild the dataset
    for index, leaderboard in top_k_leaderboard.items():
        #print(len(dataset.imgs))
        new_imgs += [tup[1] for tup in leaderboard]
        new_labels += [index for _ in leaderboard]

    dataset.filepaths = new_imgs
    dataset.labels = new_labels

    with open(filename, "wb") as f:
        pickle.dump({'filepaths':new_imgs, 'labels':new_labels}, f)

    return dataset
  
def pseudolabel_top_k(data_name, k, template, dataset, 
                      classnames, transform, clip_model,
                      label_to_idx, device):
    filename = f'pseudolabels/{data_name}_{k}_pseudolabels.pickle'
    if os.path.exists(filename):
        #print('Load pseudolabels')
        with open(filename, 'rb') as f:
            pseudolabels = pickle.load(f)
            new_imgs = pseudolabels['filepaths']
            new_labels = pseudolabels['labels']

            dataset.filepaths = new_imgs
            dataset.labels = new_labels
    else:
        dataset = compute_pseudo_labels(k, template, dataset, 
                                    classnames, transform, clip_model,
                                    label_to_idx, device, filename)

    return dataset
