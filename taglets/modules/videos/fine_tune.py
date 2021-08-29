import os
import logging
from enum import Enum

import torch

from ..module import Module
from ...pipeline import VideoTaglet


log = logging.getLogger(__name__)

class Optimizer(Enum):
    SGD = 1
    ADAM = 2

class FineTuneVideoModule(Module):
    """
    A module that fine-tunes the task's initial model.
    """
    def __init__(self, task):
        super().__init__(task)
        self.taglets = [FineTuneVideoTaglet(task, optimizer=Optimizer.SGD)]


class FineTuneVideoTaglet(VideoTaglet):
    def __init__(self, task, optimizer):
        super().__init__(task)
        self.name = 'finetune-video'   
        self.opt_type = optimizer       

        if os.getenv("LWLL_TA1_PROB_TASK") is not None:
            self.save_dir = os.path.join('/home/tagletuser/trained_models', self.name)
        else:
            self.save_dir = os.path.join('trained_models', self.name)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        output_shape = self.model.blocks[6].proj.in_features 
        self.model.blocks[6].proj = torch.nn.Linear(output_shape, len(self.task.classes))
                         
        # Parameters needed to be updated based on freezing layer
        params_to_update = []
        for param in self.model.parameters():
            if param.requires_grad:
                params_to_update.append(param)

        log.info(f"number to update: {len(params_to_update)}")
        self._params_to_update = params_to_update
        if self.opt_type == Optimizer.SGD:
            self.optimizer = torch.optim.SGD(self._params_to_update, lr=self.lr, weight_decay=1e-4, momentum=0.9)
        else:
            self.optimizer = torch.optim.Adam(self._params_to_update, lr=self.lr, weight_decay=1e-4)
        
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)


