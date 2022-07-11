from .taglet import ImageTrainable, VideoTrainable

import logging
import os
import torch
import numpy as np

log = logging.getLogger(__name__)


class EndModelMixin():
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
        os.makedirs(self.save_dir, exist_ok=True)
        self.criterion = self.soft_cross_entropy

        params_to_update = []
        for param in self.model.parameters():
            if param.requires_grad:
                params_to_update.append(param)
        self._params_to_update = params_to_update
        self.optimizer = torch.optim.Adam(self._params_to_update, lr=self.lr, weight_decay=1e-4)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.1)

    def train(self, train_data, val_data, unlabeled_data=None, num_checkpoint=None):
        if len(train_data) > 200000:
            self.num_epochs = 10
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.1)
        elif len(train_data) > 80000:
            self.num_epochs = 20
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
    
        super().train(train_data, val_data, unlabeled_data, num_checkpoint=num_checkpoint)

    @staticmethod
    def soft_cross_entropy(outputs, target):
        outputs = outputs.double()
        target = target.double()
        logs = torch.nn.LogSoftmax(dim=1)
        return torch.mean(torch.sum(-target * logs(outputs), 1))

    @staticmethod
    def _get_train_acc(outputs, labels):
        return torch.sum(torch.max(outputs, 1)[1] == torch.max(labels, 1)[1])
    

class ImageEndModel(EndModelMixin, ImageTrainable):
    """
    An end model for image data
    """
    

class VideoEndModel(EndModelMixin, VideoTrainable):
    """
    An end model for video data
    """


class RandomEndModel(ImageEndModel):
    def train(self, train_data, val_data, unlabeled_data=None):
        pass
    
    def predict(self, data):
        if len(data) == 0:
            raise ValueError('Should not get an empty dataset')
        if isinstance(data[0], tuple):
            data_loader = torch.utils.data.DataLoader(
                dataset=data, batch_size=self.batch_size, shuffle=False,
                num_workers=self.num_workers, pin_memory=True
            )
            labels = []
            for batch in data_loader:
                inputs, targets = batch
                labels.append(targets)
            labels = torch.cat(labels).numpy()
            return np.random.rand(len(data), len(self.task.classes)), labels
        else:
            return np.random.rand(len(data), len(self.task.classes))
