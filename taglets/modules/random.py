from .module import Module
from ..pipeline import ImageTaglet

# from accelerate import Accelerator
# accelerator = Accelerator()
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from datetime import timedelta
accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timedelta=timedelta(seconds=18000))]) # increased timeout limit from half an hour to 5 hours
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


class RandomTaglet(ImageTaglet):
    def train(self, train_data, val_data, unlabeled_data=None):
        pass
    
    def predict(self, data):
        if len(data) == 0:
            raise ValueError('Should not get an empty dataset')
        if isinstance(data[0], tuple):
            data_loader = self._get_dataloader(data, False)
            labels = []
            for batch in data_loader:
                inputs, targets = batch
                labels.append(accelerator.gather(targets).cpu())
            labels = torch.cat(labels).numpy()
            dataset_len = len(data_loader.dataset)
            labels = labels[:dataset_len]
            return np.random.rand(len(data), len(self.task.classes)), labels
        else:
            return np.random.rand(len(data), len(self.task.classes))
