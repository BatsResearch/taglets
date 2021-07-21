from ..module import Module
from ...pipeline import VideoTaglet

import logging
import os
import torch

log = logging.getLogger(__name__)


class BaselineVideoModule(Module):
    """
    A module that fine-tunes the task's initial model.
    """
    def __init__(self, task):
        super().__init__(task)
        self.taglets = [BaselineVideoTaglet(task)]


class BaselineVideoTaglet(VideoTaglet):
    def __init__(self, task):
        super().__init__(task)
        self.name = 'baseline-video'
        #for param in self.model.parameters():
        #    param.requires_grad = False
        
        output_shape = self.model.blocks[6].proj.in_features 
        self.model.blocks[6].proj = torch.nn.Linear(output_shape, len(self.task.classes))
                                    
        #self.model.blocks[6].proj = torch.nn.Sequential(
        #                            torch.nn.Linear(output_shape, 1056),
        #                            torch.nn.ReLU(inplace=True),
        #                            torch.nn.Linear(1056, len(self.task.classes)))
                

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

        log.info(f"number to update: {len(params_to_update)}")
        self._params_to_update = params_to_update
        self.optimizer = torch.optim.SGD(self._params_to_update, lr=self.lr, weight_decay=1e-4, momentum=0.9)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.1)