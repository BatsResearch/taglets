import os
import torch
import random
import pickle
import logging
import numpy as np
from tqdm import tqdm
import torchvision.transforms as transforms

import clip
from PIL import Image

log = logging.getLogger(__name__)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class Config(object):
    def __init__(self, config):
        for k, v in config.items():
            setattr(self, k, v)   

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
    transform = transforms.Compose(transform_fn + [preprocess_transform])
    
    return transform

def prepare_data_ssl_loss(aug_1, aug_2):
        
        # Concatenate prompts, a positive pair is concat_prompts[i] 
        # and concat_prompts[i+batch_size]
        concat_prompts = torch.cat((aug_1, aug_2), dim=0)
        # Similarity labels
        labels = torch.arange(concat_prompts.shape[0] // 2) # creates [0, ..., batch size]
        labels = torch.cat((labels, labels)) # concat labels for pairing

        return concat_prompts, labels

def pseudolabel_top_k(k, template, dataset, 
                      classnames, transform, clip_model,
                      device):
    filename = 'pseudolabels.pickle'
    if os.path.exists(filename):
        #print('Load pseudolabels')
        with open(filename, 'rb') as f:
            pseudolabels = pickle.load(f)
            new_imgs = pseudolabels['filepaths']
            new_labels = pseudolabels['labels']

            dataset.filepaths = new_imgs
            dataset.labels = new_labels

        return dataset
    else:
        prompts = [f"{template}{' '.join(i.split('_'))}" \
                            for i in classnames]
        text = clip.tokenize(prompts).to(device)
        
        #to find the top k for each class, each class has it's own "leaderboard"
        top_k_leaderboard = {i : [] for i in range(len(classnames))} #maps class idx -> (confidence, image_path) tuple
        
        log.info(f"Compute {k} pseudo-labeles")
        for i, image_path in enumerate(tqdm(dataset.filepaths)):
            img = Image.open(image_path).convert('RGB')
            img = transform(img).to(device)
            with torch.no_grad():
                logits_per_image, logits_per_text = clip_model(torch.unsqueeze(img, 0).to(device), text)
                probs = logits_per_image.softmax(dim=-1)
                idx_preds = torch.argmax(probs, dim=1)
                pred = idx_preds.item()

            """if predicted class has empty leaderboard, or if the confidence is high
            enough for predicted class leaderboard, add the new example
            """
            prob_score = probs[0][pred]
            if len(top_k_leaderboard[pred]) < k:
                    top_k_leaderboard[pred].append((prob_score, image_path))
            elif top_k_leaderboard[pred][-1][0] < prob_score: #if the confidence in predicted class "qualifies" for top-k
                    # default sorting of tuples is by first element
                    top_k_leaderboard[pred] = sorted(top_k_leaderboard[pred] + [(probs[0][pred], image_path)], reverse=True)[:k]
            else:
                #sort the other classes by confidence score
                order_of_classes = sorted([(probs[0][j], j) for j in range(len(classnames)) if j != pred], reverse=True)
                for score, index in order_of_classes:
                    if len(top_k_leaderboard[index]) < k:
                        top_k_leaderboard[index].append((probs[0][index], image_path))
                    elif top_k_leaderboard[index][-1][0] < probs[0][index]:
                        #default sorting of tuples is by first element
                        top_k_leaderboard[index] = sorted(top_k_leaderboard[index] + [((probs[0][index], image_path))], reverse=True)[:k]
        
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
