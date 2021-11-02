from .module import Module
from ..pipeline import ImageTaglet

import os
import torch


class FineTuneModule(Module):
    """
    A module that fine-tunes the task's initial model.
    """
    def __init__(self, task):
        super().__init__(task)
        self.taglets = [FineTuneTaglet(task)]


class FineTuneTaglet(ImageTaglet):
    def __init__(self, task):
        super().__init__(task)
        self.name = 'finetune'
        output_shape = self._get_model_output_shape(self.task.input_shape, self.model)
        if self.model_type == 'resnet50':
            self.model = torch.nn.Sequential(self.model,
                                             torch.nn.Linear(output_shape, len(self.task.classes)))
        else:
            self.model.fc = torch.nn.Conv2d(2048, len(self.task.classes), kernel_size=1, bias=True)
            with torch.no_grad():
                torch.nn.init.zeros_(self.model.fc.weight)
                torch.nn.init.zeros_(self.model.fc.bias)
        if os.getenv("LWLL_TA1_PROB_TASK") is not None:
            self.save_dir = os.path.join('/home/tagletuser/trained_models', self.name)
        else:
            self.save_dir = os.path.join('trained_models', self.name)
        os.makedirs(self.save_dir, exist_ok=True)

        # Parameters needed to be updated based on freezing layer
        params_to_update = []
        for param in self.model.parameters():
            if param.requires_grad:
                params_to_update.append(param)
        self._params_to_update = params_to_update
        if self.model_type == 'resnet50':
            self.optimizer = torch.optim.Adam(self._params_to_update, lr=self.lr, weight_decay=1e-4)
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.1)
        elif self.model_type == 'bigtransfer':
            self.optimizer = torch.optim.SGD(self._params_to_update, lr=0.003, momentum=0.9)
            self.lr_scheduler = None
            self.num_epochs = 500

