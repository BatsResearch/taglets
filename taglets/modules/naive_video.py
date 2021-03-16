from .module import Module
from ..pipeline import Taglet

import os
import torch


class NaiveVideoModule(Module):
    """
    A module that fine-tunes the task's initial model.
    """
    def __init__(self, task):
        super().__init__(task)
        self.taglets = [NaiveVideoTaglet(task)]


class NaiveVideoTaglet(Taglet):
    def __init__(self, task):
        super().__init__(task)
        self.name = 'naive-video'
        output_shape = self._get_model_output_shape(self.task.input_shape, self.model)
        self.model = torch.nn.Sequential(self.model,
                                         torch.nn.Linear(output_shape, len(self.task.classes)))
        if os.getenv("LWLL_TA1_PROB_TASK") is not None:
            self.save_dir = os.path.join('/home/tagletuser/trained_models', self.name)
        else:
            self.save_dir = os.path.join('trained_models', self.name)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # Parameters needed to be updated based on freezing layer
        params_to_update = []
        for param in self.model.parameters():
            if param.requires_grad:
                params_to_update.append(param)
        self._params_to_update = params_to_update
        self.optimizer = torch.optim.Adam(self._params_to_update, lr=self.lr, weight_decay=1e-4)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.1)
        
    def _train_epoch(self, rank, train_data_loader, unlabeled_data_loader=None):
        """
        Train for one epoch.
        :param train_data_loader: A dataloader containing training videos
        :param use_gpu: Whether or not to use the GPU
        :return: None
        """
        self.model.train()
        running_loss = 0
        running_acc = 0
        for batch in train_data_loader:
            inputs = batch[0]
            labels = batch[1]
            if self.use_gpu:
                inputs = inputs.cuda(rank)
                labels = labels.cuda(rank)
            num_videos = inputs.size(0)
            num_frames = inputs.size(1)
            inputs = inputs.flatten(start_dim=0, end_dim=1)

            self.optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = self.model(inputs)
                aggregated_outputs = torch.mean(outputs.view(num_videos, num_frames, -1), dim=1)
                loss = self.criterion(aggregated_outputs, labels)
                loss.backward()
                self.optimizer.step()

            running_loss += loss.item()
            running_acc += self._get_train_acc(aggregated_outputs, labels)

        if not len(train_data_loader.dataset):
            return 0, 0

        epoch_loss = running_loss / len(train_data_loader.dataset)
        epoch_acc = running_acc.item() / len(train_data_loader.dataset)

        return epoch_loss, epoch_acc
    
    def _validate_epoch(self, rank, val_data_loader,):
        """
        Train for one epoch.
        :param train_data_loader: A dataloader containing training videos
        :param use_gpu: Whether or not to use the GPU
        :return: None
        """
        self.model.eval()
        running_loss = 0
        running_acc = 0
        for batch in val_data_loader:
            inputs = batch[0]
            labels = batch[1]
            if self.use_gpu:
                inputs = inputs.cuda(rank)
                labels = labels.cuda(rank)
            num_videos = inputs.size(0)
            num_frames = inputs.size(1)
            inputs = inputs.flatten(start_dim=0, end_dim=1)

            with torch.set_grad_enabled(False):
                outputs = self.model(inputs)
                aggregated_outputs = torch.mean(outputs.view(num_videos, num_frames, -1), dim=1)
                loss = self.criterion(aggregated_outputs, labels)
                _, preds = torch.max(aggregated_outputs, 1)

            running_loss += loss.item()
            running_acc += self._get_train_acc(aggregated_outputs, labels)

        epoch_loss = running_loss / len(val_data_loader.dataset)
        epoch_acc = running_acc.item() / len(val_data_loader.dataset)

        return epoch_loss, epoch_acc

