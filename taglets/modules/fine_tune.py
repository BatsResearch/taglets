from .module import Module
from ..pipeline import Taglet

import os
import torch


class FineTuneModule(Module):
    """
    A module that fine-tunes a model pre-trained on ImageNet 1k.
    """
    def __init__(self, task):
        super().__init__(task)
        self.taglets = [FineTuneTaglet(task)]


class FineTuneTaglet(Taglet):
    def __init__(self, task):
        super().__init__(task)
        self.name = 'finetune'
        self.num_epochs = 5
        self.save_dir = os.path.join('trained_models', self.name)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def execute(self, unlabeled_data_loader, use_gpu):
        """
        Execute the Taglet on unlabeled images.
        :param unlabeled_data_loader: A dataloader containing unlabeled data
        :param use_gpu: Whether or not the use the GPU
        :return: A list of predicted labels
        """
        self.model.eval()
        if use_gpu:
            self.model = self.model.cuda()
        else:
            self.model = self.model.cpu()

        predicted_labels = []
        for inputs, index in unlabeled_data_loader:
            if use_gpu:
                inputs = inputs.cuda()
                index = index.cuda()
            with torch.set_grad_enabled(False):
                for data, ix in zip(inputs, index):
                    data = torch.unsqueeze(data, dim=0)
                    outputs = self.model(data)
                    _, preds = torch.max(outputs, 1)
                    predicted_labels.append(preds.item())
        return predicted_labels
