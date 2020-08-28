from .taglet import Trainable

import os
import torch


class EndModel(Trainable):
    def __init__(self, task):
        super().__init__(task)
        self.name = 'end model'
        m = torch.nn.Sequential(*list(self.model.children())[:-1])
        output_shape = self._get_model_output_shape(self.task.input_shape, m)
        self.model.fc = torch.nn.Sequential(torch.nn.Dropout(0.3),
                                            torch.nn.Linear(output_shape, len(self.task.classes)))
        self.save_dir = os.path.join('trained_models', self.name)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.criterion = self.soft_cross_entropy

    @staticmethod
    def soft_cross_entropy(outputs, target):
        outputs = outputs.double()
        target = target.double()
        logs = torch.nn.LogSoftmax(dim=1)
        return torch.mean(torch.sum(-target * logs(outputs), 1))

    @staticmethod
    def _get_train_acc(outputs, labels):
        return torch.sum(torch.max(outputs, 1)[1] == torch.max(labels, 1)[1])

