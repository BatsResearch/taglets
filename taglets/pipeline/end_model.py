from .taglet import Trainable

import os
import torch
import numpy as np


class EndModel(Trainable):
    def __init__(self, task):
        super().__init__(task)
        self.name = 'end model'
        m = torch.nn.Sequential(*list(self.model.children())[:-1])
        output_shape = self._get_model_output_shape(self.task.input_shape, m)
        self.model.fc = torch.nn.Sequential(torch.nn.Dropout(0.3),
                                            torch.nn.Linear(output_shape, len(self.task.classes)))
        if os.getenv("LWLL_TA1_PROB_TASK") is not None:
            self.save_dir = os.path.join('/home/tagletuser/trained_models', self.name)
        else:
            self.save_dir = os.path.join('trained_models', self.name)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.criterion = self.soft_cross_entropy
        self.weak_label_weight = 0.5

    def soft_cross_entropy(self, outputs, target, using_gold_labels):
        outputs = outputs.double()
        target = target.double()
        logs = torch.nn.LogSoftmax(dim=1)
        return torch.mean(
            torch.sum(-target * logs(outputs), 1) *
            np.where(using_gold_labels, 1, self.weak_label_weight)
        )

    @staticmethod
    def _get_train_acc(outputs, labels):
        return torch.sum(torch.max(outputs, 1)[1] == torch.max(labels, 1)[1])

    def _train_epoch(self, rank, train_data_loader):
        """
        Train for one epoch.
        :param train_data_loader: A dataloader containing training data
        :param use_gpu: Whether or not to use the GPU
        :return: None
        """
        self.model.train()
        running_loss = 0
        running_acc = 0
        for batch in train_data_loader:
            inputs = batch[0]
            labels = batch[1]
            using_gold_labels = batch[2]
            if self.use_gpu:
                inputs = inputs.cuda(rank)
                labels = labels.cuda(rank)

            self.optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels, using_gold_labels)
                loss.backward()
                self.optimizer.step()

            running_loss += loss.item()
            running_acc += self._get_train_acc(outputs, labels)

        if not len(train_data_loader.dataset):
            return 0, 0

        epoch_loss = running_loss / len(train_data_loader.dataset)
        epoch_acc = running_acc.item() / len(train_data_loader.dataset)

        return epoch_loss, epoch_acc