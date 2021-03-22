from .module import Module
from ..pipeline import Taglet

import logging
import numpy as np
import torch

log = logging.getLogger(__name__)


class RandomModule(Module):
    """
    A module that fine-tunes the task's initial model.
    """
    
    def __init__(self, task):
        super().__init__(task)
        self.taglets = [RandomTaglet(task)]


class RandomTaglet(Taglet):
    def train(self, train_data, val_data, unlabeled_data=None):
        pass
    
    def predict(self, data):
        if len(data) == 0:
            raise ValueError('Should not get an empty dataset')
        if isinstance(data[0], list):
            data_loader = torch.utils.data.DataLoader(
                dataset=data, batch_size=self.batch_size, shuffle=False,
                num_workers=self.num_workers, pin_memory=True
            )
            labels = []
            for batch in data_loader:
                inputs, targets = batch
                labels.append(targets)
            labels = torch.cat(labels).numpy()
            return np.random.rand(len(data), self.task.classes), labels
        else:
            return np.random.rand(len(data), self.task.classes)
