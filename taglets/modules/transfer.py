import os
import random
import torch
import logging
import numpy as np
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import Subset

from ..data.custom_dataset import CustomImageDataset
from .module import Module
from ..pipeline import Cache, ImageTaglet
from ..scads import Scads, ScadsEmbedding

log = logging.getLogger(__name__)


class TransferModule(Module):
    """
    A module that pre-trains on datasets selected from the SCADS and then
    transfers to available labeled data
    """
    def __init__(self, task):
        super().__init__(task)
        self.taglets = [TransferTaglet(task)]


class TransferTaglet(ImageTaglet):
    def __init__(self, task, freeze=False, is_norm=False):
        super().__init__(task)
        self.name = 'transfer'
        if os.getenv("LWLL_TA1_PROB_TASK") is not None:
            self.save_dir = os.path.join('/home/tagletuser/trained_models', self.name)
        else:
            self.save_dir = os.path.join('trained_models', self.name)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            
        self.freeze = freeze
        self.is_norm = is_norm
        self.img_per_related_class = 600 if not os.environ.get("CI") else 1
        self.num_related_class = 5

    def transform_image(self, train=True):
        """
        Get the transform to be used on an image.
        :return: A transform
        """
        data_mean = [0.485, 0.456, 0.406]
        data_std = [0.229, 0.224, 0.225]
    
        if train:
            return transforms.Compose([
                transforms.RandomResizedCrop(self.task.input_shape, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=data_mean, std=data_std)
            ])
        else:
            return transforms.Compose([
                transforms.Resize(self.task.input_shape),
                transforms.ToTensor(),
                transforms.Normalize(mean=data_mean, std=data_std)
            ])

    def _get_scads_data(self):
        data = Cache.get("scads", self.task.classes)
        if data is not None:
            image_paths, image_labels, all_related_class = data
        else:
            root_path = Scads.get_root_path()
            Scads.open(self.task.scads_path)
            ScadsEmbedding.load(self.task.scads_embedding_path)
            image_paths = []
            image_labels = []
            visited = set()

            def get_images(node, label):
                if node.get_conceptnet_id() not in visited:
                    visited.add(node.get_conceptnet_id())
                    images = node.get_images_whitelist(self.task.whitelist)
                    if len(images) < self.img_per_related_class:
                        return False
                    images = random.sample(images, self.img_per_related_class)
                    images = [os.path.join(root_path, image) for image in images]
                    image_paths.extend(images)
                    image_labels.extend([label] * len(images))
                    log.debug("Source class found: {}".format(node.get_conceptnet_id()))
                    return True
                return False

            all_related_class = 0
            for conceptnet_id in self.task.classes:
                cur_related_class = 0
                target_node = Scads.get_node_by_conceptnet_id(conceptnet_id)
                if get_images(target_node, all_related_class):
                    cur_related_class += 1
                    all_related_class += 1

                neighbors = ScadsEmbedding.get_related_nodes(target_node, self.num_related_class * 100)
                for neighbor in neighbors:
                    if get_images(neighbor, all_related_class):
                        cur_related_class += 1
                        all_related_class += 1
                        if cur_related_class >= self.num_related_class:
                            break

            Scads.close()
            Cache.set('scads', self.task.classes,
                      (image_paths, image_labels, all_related_class))

        transform = self.transform_image(train=True)
        train_val_data = CustomImageDataset(image_paths,
                                            labels=image_labels,
                                            transform=transform)

        # 80% for training, 20% for validation
        train_percent = 0.8
        num_data = len(train_val_data)
        indices = list(range(num_data))
        train_split = int(np.floor(train_percent * num_data))
        np.random.shuffle(indices)
        train_idx = indices[:train_split]
        valid_idx = indices[train_split:]

        train_dataset = Subset(train_val_data, train_idx)
        val_dataset = Subset(train_val_data, valid_idx)

        return train_dataset, val_dataset, all_related_class

    def _set_num_classes(self, num_classes):
        m = torch.nn.Sequential(*list(self.model.children())[:-1])
        output_shape = self._get_model_output_shape(self.task.input_shape, m)
        self.model.fc = NormLinear(torch.nn.Linear(output_shape, num_classes), self.is_norm)

        params_to_update = []
        for param in self.model.parameters():
            if param.requires_grad:
                params_to_update.append(param)
        self._params_to_update = params_to_update
        self.optimizer = torch.optim.Adam(self._params_to_update, lr=self.lr, weight_decay=1e-4)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.1)

    def train(self, train_data, val_data, unlabeled_data=None):
        scads_train_data, scads_val_data, scads_num_classes = self._get_scads_data()
        log.info("Source classes found: {}".format(scads_num_classes))
        
        if scads_num_classes == 0:
            self.valid = False
            return

        orig_num_epochs = self.num_epochs
        self.num_epochs = 5 if not os.environ.get("CI") else 5
        self._set_num_classes(scads_num_classes)
        super(TransferTaglet, self).train(scads_train_data, scads_val_data, unlabeled_data)
        self.num_epochs = orig_num_epochs

        # Freeze layers
        if self.freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        orig_num_epochs = self.num_epochs
        self.num_epochs = 25 if not os.environ.get("CI") else 5
        self._set_num_classes(len(self.task.classes))
        super(TransferTaglet, self).train(train_data, val_data, unlabeled_data)
        self.num_epochs = orig_num_epochs

        # Unfreeze layers
        if self.freeze:
            for param in self.model.parameters():
                param.requires_grad = True

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

