import torch
import json
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
    UniformCropVideo
)
from typing import Dict
import random

####################
# SlowFast transform
####################

# We might want different options for transforming the data depending on the model we have

side_size = 256
mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]
crop_size = 256
num_frames = 32
sampling_rate = 2
frames_per_second = 30
alpha = 4

class PackPathway(torch.nn.Module):
    """
    Transform for converting video frames as a list of tensors.
    """
    def __init__(self):
        super().__init__()

    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // alpha
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list

transform =  ApplyTransformToKey(
    key="video",
    transform=Compose(
        [
            UniformTemporalSubsample(num_frames),
            Lambda(lambda x: x/255.0),
            NormalizeVideo(mean, std),
            ShortSideScale(
                size=side_size
            ),
            CenterCropVideo(crop_size),
            PackPathway()
        ]
    ),
)


import os

from torch.utils.data import Dataset
from PIL import Image
import torch
import torchvision.transforms as transforms

class CustomImageDataset(Dataset):
    """
    A custom dataset used to create dataloaders.
    """
    def __init__(self, filepaths, labels=None, label_map=None, transform=None):
        """
        Create a new CustomImageDataset.
        
        :param filepaths: A list of filepaths. 
        :param labels: A list of labels
        :param label_map: A dictionary to map string labels to intergers
        :param transform: A transform to perform on the images
        """
        self.filepaths = filepaths
        self.labels = labels
        self.label_map = label_map
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.filepaths[index]).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.labels is not None:
            if self.label_map is not None:
                label = torch.tensor(self.label_map[(self.labels[index])])
            else:
                label = torch.tensor(int(self.labels[index]))
            return img, label
        else:
            return img

    def __len__(self):
        return len(self.filepaths)
    

class CustomVideoDataset(Dataset):
    """
        A custom dataset used to create dataloaders.
        """
    
    def __init__(self, filepaths, n_frames=32, labels=None, label_map=None, transform=None, clips_dictionary=None):
        """
        Create a new CustomVideoDataset.
        :param filepaths: A list of filepaths.
        :param labels: A list of labels
        :param label_map: A dictionary to map string labels to intergers
        :param transform: A transform to perform on the frames
        :pram clips_dictionary: dictionary (id clip, list images) to get frames of a clip
        """
        self.filepaths = filepaths
        self.labels = labels
        self.label_map = label_map
        self.transform = transform
        self.clips_dictionary = clips_dictionary
        self.n_frames = n_frames
    
    def __getitem__(self, index):
        clip_id = str(os.path.basename(self.filepaths[index]))#int(os.path.basename(self.filepaths[index]))  # chech what path you have/want
        frames_paths = self.clips_dictionary[clip_id]
        # print(f"FRAMES list[:2]: {frames_paths[:2]} and number of frames {len(frames_paths)}")
        
        frames = []
        sample_frames = frames_paths#random.choices(frames_paths, k=self.n_frames)
        for f in sample_frames:#frames_paths:  # get same size clips - random pick for eval
            frame = Image.open(f).convert('RGB')
            
            if self.transform is not None:  # BE CAREFUL TRANSFORMATION MIGHT NEED TO CHANGE FOR VIDEO EVAL!!!!!
                frame = self.transform(frame)
            frames.append(frame)
        
        img = torch.stack(frames) # need to be of the same size!
        img = torch.transpose(img, 0, 1) 
        video_data = {'video':img}
        img = transform(video_data)


        if self.labels is not None:
            if self.label_map is not None:
                label = torch.tensor(self.label_map[(self.labels[index])])
            else:
                label = torch.tensor(int(self.labels[index]))
            return img, label
        else:
            return img
    
    def __len__(self):
        return len(self.filepaths)


class SoftLabelDataset(Dataset):
    """
    A custom dataset used to create dataloaders.
    """
    def __init__(self, dataset, labels, remove_old_labels=False):
        """
        Create a new SoftLabelDataset.
        :param dataset: A PyTorch dataset
        :param labels: A list of labels
        :param remove_old_labels: A boolean indicating whether to the dataset returns labels that we do not use
        """
        self.dataset = dataset
        self.labels = labels
        self.remove_old_labels = remove_old_labels

    def __getitem__(self, index):
        data = self.dataset[index]
        label = self.labels[index]
        
        if self.remove_old_labels:
            data = data[0]
            
        return data, label

    def __len__(self):
        return len(self.dataset)
    
def transform_image(train=True):
    """
    Get the transform to be used on an image.
    :return: A transform
    """
    data_mean = [0.485, 0.456, 0.406]
    data_std = [0.229, 0.224, 0.225]
    # Remember to check it for video and eval
    if train:
        return transforms.Compose([
            #transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            #transforms.Normalize(mean=data_mean, std=data_std)
        ])
    else:
        return transforms.Compose([
            #transforms.Resize((224, 224)),
            transforms.ToTensor(),
            #transforms.Normalize(mean=data_mean, std=data_std)
        ])

# The duration of the input clip is also specific to the model.
clip_duration = (num_frames * sampling_rate)/frames_per_second

# Device on which to run the model
# Set to cuda to load on GPU
device = "cpu"

# Pick a pretrained model and load the pretrained weights
model_name = "slowfast_r50"
model = torch.hub.load("facebookresearch/pytorchvideo", model=model_name, pretrained=True)

# Set to eval mode and move to desired device
model = model.to(device)
model = model.eval()


## Finetune try 

import os
import requests

import torch.nn as nn

feature_extract = True
num_classes = 51

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

# Input required by the modelssssss

# define function where I can initialize each of the model we might want to use
model.blocks[6].proj = nn.Linear(2304, num_classes)


## API INTERACTION

url = 'https://api-dev.lollllz.com'
secret = 'a5aed2a8-db80-4b22-bf72-11f2d0765572'
headers = {'user_secret': secret, 'govteam_secret': os.environ.get('GOVTEAM_SECRET')}

# This is a convenience for development purposes, IN EVAL ALWAYS USE `full`
data_type = 'sample' # can either be `sample` or `full`

r = requests.post(f"{url}/auth/create_session", json={'session_name': 'testing', 'data_type': data_type, 'task_id': 'problem_test_video_classification'}, headers=headers)
r.json()
session_token = r.json()['session_token']

headers = {'user_secret': secret, 'session_token': session_token, 'govteam_secret': os.environ.get('GOVTEAM_SECRET')}

r = requests.get(f"{url}/seed_labels", headers=headers)
print(json.dumps(r.json(), indent = 4))

r_data = requests.get(f"{url}/dataset_metadata/hmdb", headers=headers)

#### 
import numpy as np
root = '../../../../../lwll/development/hmdb/'

unlabeled_image_path = '../../../../../lwll/development/hmdb/hmdb_full/train'
video = True


labels = r.json()['Labels']

labels_list = []
dictionary_clips = {}
for clip in labels:
    #action_frames = [str(i)+'.jpg' for i in range(clip['start_frame'], clip['end_frame'])]
    action_frames = [str(clip['id']) + '/' + str(i)+'.jpg' for i in range(clip['start_frame'], clip['end_frame'])]
    dictionary_clips[clip["id"]] = action_frames
    labels_list.append([clip["class"], clip["id"]])

image_labels, image_names  = list(zip(*labels_list))

image_paths = [os.path.join(unlabeled_image_path, str(image_name)) for image_name in image_names]



if video:
    paths_dictionary_clips = {}
    for clip, frames in dictionary_clips.items():
        paths_dictionary_clips[clip] = [os.path.join(unlabeled_image_path, str(f)) for f in frames]
    dictionary_clips = paths_dictionary_clips
else:
    dictionary_clips = None

image_paths = np.asarray(image_paths)
image_labels = np.asarray(image_labels)


checkpoint_num = 1

if checkpoint_num >= 4:
    # 80% for training, 20% for validation
    train_percent = 0.8
    num_data = len(image_paths)
    indices = list(range(num_data))
    train_split = int(np.floor(train_percent * num_data))
    np.random.shuffle(indices)
    train_idx = indices[:train_split]
    val_idx = indices[train_split:]
else:
    train_idx = list(range(len(image_paths)))
    val_idx = []

label_map = {}
class_names = r_data.json()['dataset_metadata']['classes']
for idx, item in enumerate(class_names):
    label_map[item] = idx

label_map = label_map


if video:
    train_dataset = CustomVideoDataset(image_paths[train_idx],
                                       labels=image_labels[train_idx],
                                       label_map=label_map,
                                       transform=transform_image(train=True),
                                       clips_dictionary=dictionary_clips)
    if len(val_idx) != 0:
        val_dataset = CustomVideoDataset(image_paths[val_idx],
                                         labels=image_labels[val_idx],
                                         label_map=label_map,
                                         transform=transform_image(train=False),
                                         clips_dictionary=dictionary_clips)
    else:
        val_dataset = None



# Move the inputs to the desired device
inputs = train_dataset[0][0]["video"]
inputs = [i.to(device)[None, ...] for i in inputs]

# Pass the input clip through the model
preds = model(inputs)

print(preds)

# Get the predicted classes
post_act = torch.nn.Softmax(dim=1)
preds = post_act(preds)
pred_classes = preds.topk(k=5).indices

# Map the predicted classes to the label names
map_label = {idx:c for c,idx in label_map.items()}
pred_class_names = [map_label[int(i)] for i in pred_classes[0]]
print("Predicted labels: %s" % ", ".join(pred_class_names))