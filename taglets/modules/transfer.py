import copy
import os
import torch
import logging
import torchvision.transforms as transforms
import torch.nn as nn
from accelerate import Accelerator
accelerator = Accelerator()

from .module import Module
from ..pipeline import ImageTagletWithAuxData, Cache

log = logging.getLogger(__name__)


class TransferModule(Module):
    """
    A module that pre-trains on datasets selected from the SCADS and then
    transfers to available labeled data
    """
    def __init__(self, task):
        super().__init__(task)
        self.taglets = [TransferTaglet(task)]


class TransferTaglet(ImageTagletWithAuxData):
    def __init__(self, task, freeze=False, is_norm=False):
        super().__init__(task)
        self.name = 'transfer'
        if os.getenv("LWLL_TA1_PROB_TASK") is not None:
            self.save_dir = os.path.join('/home/tagletuser/trained_models', self.name)
        else:
            self.save_dir = os.path.join('trained_models', self.name)
        os.makedirs(self.save_dir, exist_ok=True)
            
        self.freeze = freeze
        self.is_norm = is_norm

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

    def _set_num_classes(self, num_classes):
        m = torch.nn.Sequential(*list(self.model.children())[:-1])
        output_shape = self._get_model_output_shape(self.task.input_shape, m)
        self.model.fc = NormLinear(torch.nn.Linear(output_shape, num_classes), self.is_norm)

        params_to_update = []
        for param in self.model.parameters():
            if param.requires_grad:
                params_to_update.append(param)
        self._params_to_update = params_to_update
        self.optimizer = torch.optim.SGD(self._params_to_update, lr=0.003, momentum=0.9)
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[20, 30], gamma=0.1)

    def train(self, train_data, val_data, unlabeled_data=None):
        aux_weights = Cache.get("scads-weights", self.task.classes)
        if aux_weights is None:
            scads_train_data, scads_num_classes = self._get_scads_data()
            log.info("Source classes found: {}".format(scads_num_classes))
            
            if scads_num_classes == 0:
                self.valid = False
                return
    
            orig_num_epochs = self.num_epochs
            self.num_epochs = 5 if not os.environ.get("CI") else 5
            self._set_num_classes(scads_num_classes)
            super(TransferTaglet, self).train(scads_train_data, None, None)
            self.num_epochs = orig_num_epochs
            
            self.model.fc = nn.Identity()
            aux_weights = copy.deepcopy(self.model.state_dict())
            Cache.set('scads-weights', self.task.classes, aux_weights)
        self.model.load_state_dict(aux_weights, strict=False)

        # Freeze layers
        if self.freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        orig_num_epochs = self.num_epochs
        if os.environ.get("CI"):
            self.num_epochs = 5
        elif len(train_data) > 200000:
            self.num_epochs = 5
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[3, 4], gamma=0.1)
        elif len(train_data) > 100000:
            self.num_epochs = 10
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[6, 8], gamma=0.1)
        elif len(train_data) > 50000:
            self.num_epochs = 20
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[12, 16], gamma=0.1)
        elif len(train_data) > 25000:
            self.num_epochs = 30
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[18, 24], gamma=0.1)
        else:
            self.num_epochs = 40
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[20, 30], gamma=0.1)
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

