import os
import copy
import random
import torch
import logging
import numpy as np
import torch.nn as nn


import torchvision.transforms as transforms
import pytorchvideo.transforms as video_transform
import torchvision.transforms._transforms_video as transform_video
from accelerate import Accelerator
accelerator = Accelerator()

from ..module import Module
from ...pipeline import VideoTagletWithAuxData, Cache


log = logging.getLogger(__name__)

def normalize(x):
    norm = x.pow(2).sum(1, keepdim=True).pow(1. / 2)
    out = x.div(norm)
    return out

class NormLinear(nn.Module):
    def __init__(self, model, is_norm):
        super().__init__()
        self.model = model
        self.is_norm = is_norm

    def forward(self, x):
        if self.is_norm:
            x = normalize(x)
        x = self.model(x)
        return x

class PackPathway(torch.nn.Module):
    """
    Transform for converting video frames as a list of tensors.
    """
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // self.alpha
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list


class TransferVideoModule(Module):
    """
    A module that pre-trains on datasets selected from the SCADS and then
    transfers to available labeled data
    """
    def __init__(self, task):
        super().__init__(task)
        self.taglets = [TransferVideoTaglet(task)]

class TransferVideoTaglet(VideoTagletWithAuxData):
    def __init__(self, task, video=True, freeze=False, is_norm=False):
        super().__init__(task)
        self.name = 'transfer-video'
        self.video = video
        if os.getenv("LWLL_TA1_PROB_TASK") is not None:
            self.save_dir = os.path.join('/home/tagletuser/trained_models', self.name)
        else:
            self.save_dir = os.path.join('trained_models', self.name)
        os.makedirs(self.save_dir, exist_ok=True)
            
        self.freeze = freeze
        self.is_norm = is_norm
        self.output_shape = self.model.blocks[6].proj.in_features 

    def transform_image(self, train=True, video=False):
        """
        Get the transform to be used on an image.
        :return: A transform
        """
        data_mean = [0.485, 0.456, 0.406]
        data_std = [0.229, 0.224, 0.225]
        # Remember to check it for video and eval
        if video:
            return transforms.Compose([transforms.ToTensor()])
        
        if train:
            return transforms.Compose([
                transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=data_mean, std=data_std)
            ])
        else:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=data_mean, std=data_std)
            ])

    def transformer_video(self):   
        """Trasformation valid for SlowFast"""
        side_size = 256
        mean = [0.45, 0.45, 0.45]
        std = [0.225, 0.225, 0.225]
        crop_size = 256
        num_frames = 32
        alpha = 4

        
        return  video_transform.ApplyTransformToKey(key="video",
                                    transform=transforms.Compose([
                                        video_transform.UniformTemporalSubsample(num_frames),
                                        #transforms.Lambda(lambda x: x/255.0),
                                        transform_video.NormalizeVideo(mean, std),# transform_video
                                        video_transform.ShortSideScale(size=side_size),
                                        transform_video.CenterCropVideo(crop_size),# transform_video
                                        PackPathway(alpha)
                                    ])
                                    )

    def _set_num_classes(self, num_classes, aux=False):
        #output_shape = self.model.blocks[6].proj.in_features 
        #self.model.blocks[6].proj = NormLinear(torch.nn.Linear(self.output_shape, num_classes), self.is_norm)
        self.model.blocks[6].proj = torch.nn.Linear(self.output_shape, num_classes)

        params_to_update = []
        for param in self.model.parameters():
            if param.requires_grad:
                params_to_update.append(param)
        self._params_to_update = params_to_update
        
        if aux:
            self.optimizer = torch.optim.SGD(self._params_to_update, lr=0.001, weight_decay=1e-4, momentum=0.9)
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        else:
            self.optimizer = torch.optim.SGD(self._params_to_update, lr=self.lr, weight_decay=1e-4, momentum=0.9)
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
    
    def train(self, train_data, val_data, unlabeled_data=None):
        aux_weights = Cache.get("scads-weights", self.task.classes)
        if aux_weights is None:
            scads_train_data, scads_val_data, scads_num_classes = self._get_scads_data(split=False)
            log.info("Source classes found: {}".format(scads_num_classes))
            
            if scads_num_classes == 0:
                self.valid = False
                return
    
            orig_num_epochs = self.num_epochs
            self.num_epochs = 10 if not os.environ.get("CI") else 5
            self._set_num_classes(scads_num_classes, aux=True)
            super(TransferVideoTaglet, self).train(scads_train_data, scads_val_data, None) # Add here validation data return validation set from _get_scads_data()
            self.num_epochs = orig_num_epochs
            
            self.model.blocks[6].proj = nn.Sequential()
            aux_weights = copy.deepcopy(self.model.state_dict())
            Cache.set('scads-weights', self.task.classes, aux_weights)
        self.model.load_state_dict(aux_weights, strict=False)

        # Freeze layers
        if self.freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        orig_num_epochs = self.num_epochs
        if self.task.checkpoint == 7:
            self.num_epochs = 10 if not os.environ.get("CI") else 5
        elif 4 <= self.task.checkpoint <= 6:
            self.num_epochs = 30 if not os.environ.get("CI") else 5
        elif self.task.checkpoint == 0:
            self.num_epochs = 50 if not os.environ.get("CI") else 5
        else:
            self.num_epochs = 40 if not os.environ.get("CI") else 5
        self._set_num_classes(len(self.task.classes))
        super(TransferVideoTaglet, self).train(train_data, val_data, unlabeled_data)
        self.num_epochs = orig_num_epochs

        # Unfreeze layers
        if self.freeze:
            for param in self.model.parameters():
                param.requires_grad = True
    
