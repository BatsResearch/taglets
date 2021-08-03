import os
import torch
import logging
import numpy as np
import torchvision.transforms as transforms
import torch.nn as nn
from accelerate import Accelerator
accelerator = Accelerator()

from .module import Module
from ..pipeline import ImageTagletWithAuxData

log = logging.getLogger(__name__)


class MultiTaskModel(nn.Module):
    def __init__(self, model, num_target, num_source, input_shape):
        super().__init__()
        self.num_target= num_target
        self.num_source = num_source
        self.base = nn.Sequential(*list(model.children())[:-1])
        output_shape = self._get_model_output_shape(input_shape, self.base)
        self.fc_target = nn.Conv2d(2048, self.num_target, kernel_size=1, bias=True)
        self.fc_source = nn.Conv2d(2048, self.num_source, kernel_size=1, bias=True)
        with torch.no_grad():
            torch.nn.init.zeros_(self.fc_target.weight)
            torch.nn.init.zeros_(self.fc_target.bias)
            torch.nn.init.zeros_(self.fc_source.weight)
            torch.nn.init.zeros_(self.fc_source.bias)

    def forward(self, target_inputs, source_inputs=None):
        x = self.base(target_inputs)
        target_outputs = self.fc_target(x)[...,0,0]
        if source_inputs is None:
            return target_outputs
        else:
            x = self.base(source_inputs)
            source_outputs = self.fc_source(x)[...,0,0]
            return target_outputs, source_outputs

    def _get_model_output_shape(self, in_size, mod):
        """
        Adopt from https://gist.github.com/lebedov/0db63ffcd0947c2ea008c4a50be31032
        Compute output size of Module `mod` given an input with size `in_size`
        :param in_size: input shape (height, width)
        :param mod: PyTorch model
        :return:
        """
        mod = mod.cpu()
        f = mod(torch.rand(2, 3, *in_size))
        return int(np.prod(f.size()[1:]))


class MultiTaskModule(Module):
    """
    A module that pre-trains on datasets selected from the SCADS and then
    transfers to available labeled data
    """
    def __init__(self, task):
        super().__init__(task)
        self.taglets = [MultiTaskTaglet(task)]


class MultiTaskTaglet(ImageTagletWithAuxData):
    def __init__(self, task):
        super().__init__(task)
        self.name = 'multitask'
        self.num_epochs = 8 if not os.environ.get("CI") else 5
        if os.getenv("LWLL_TA1_PROB_TASK") is not None:
            self.save_dir = os.path.join('/home/tagletuser/trained_models', self.name)
        else:
            self.save_dir = os.path.join('trained_models', self.name)
        os.makedirs(self.save_dir, exist_ok=True)
        self.source_data = None
        
        self.batch_size = self.batch_size // 2

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

    def train(self, train_data, val_data, unlabeled_data=None):
        # Get Scads data and set up model
        self.source_data, num_classes = self._get_scads_data()
        log.info("Source classes found: {}".format(num_classes))
        log.info("Number of source training images: {}".format(len(self.source_data)))
        
        if num_classes == 0:
            self.valid = False
            return

        self.model = MultiTaskModel(self.model, len(self.task.classes),
                                    num_classes, self.task.input_shape)

        params_to_update = []
        for param in self.model.parameters():
            if param.requires_grad:
                params_to_update.append(param)
        self._params_to_update = params_to_update
        self.optimizer = torch.optim.SGD(self._params_to_update, lr=0.003, momentum=0.9)
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[4, 6], gamma=0.1)
        
        if len(train_data) < 1024:
            num_duplicates = (1024 // len(train_data)) + 1
            train_data = torch.utils.data.ConcatDataset([train_data] * num_duplicates)
        
        super(MultiTaskTaglet, self).train(train_data, val_data, unlabeled_data)

    def _do_train(self, train_data, val_data, unlabeled_data=None):
        self.source_data_loader = self._get_dataloader(data=self.source_data, shuffle=True)
        super(MultiTaskTaglet, self)._do_train(train_data, val_data, unlabeled_data)

    def _train_epoch(self, train_data_loader, unlabeled_train_loader=None):
        self.model.train()
        running_loss = 0
        running_acc = 0
        total_len = 0
        data_iter = iter(train_data_loader)
        for source_batch in self.source_data_loader:
            try:
                target_batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_data_loader)
                target_batch = next(data_iter)

            source_inputs, source_labels = source_batch
            target_inputs, target_labels = target_batch

            self.optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = self.model(target_inputs, source_inputs)
                target_outputs, source_outputs = outputs
                source_loss = self.criterion(source_outputs, source_labels)
                target_loss = self.criterion(target_outputs, target_labels)
                loss = source_loss + target_loss

                accelerator.backward(loss)
                self.optimizer.step()

            source_outputs = accelerator.gather(source_outputs.detach())
            source_labels = accelerator.gather(source_labels)
            target_outputs = accelerator.gather(target_outputs.detach())
            target_labels = accelerator.gather(target_labels)

            running_loss += loss.item()
            running_acc += self._get_train_acc(source_outputs, source_labels).item()
            running_acc += self._get_train_acc(target_outputs, target_labels).item()
            total_len += len(source_labels)
            total_len += len(target_labels)

        if not len(train_data_loader):
            return 0, 0

        epoch_loss = running_loss / len(train_data_loader)
        epoch_acc = running_acc / total_len

        return epoch_loss, epoch_acc
