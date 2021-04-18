from .module import Module
from ..pipeline import VideoTaglet

import logging
import os
import torch

log = logging.getLogger(__name__)


class NaiveVideoModule(Module):
    """
    A module that fine-tunes the task's initial model.
    """
    def __init__(self, task):
        super().__init__(task)
        self.taglets = [NaiveVideoTaglet(task)]


class NaiveVideoTaglet(VideoTaglet):
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
