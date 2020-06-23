from .module import Module
from ..pipeline import Taglet

import os
import torch


class FineTuneModule(Module):
    """
    A module that fine-tunes the task's initial model.
    """
    def __init__(self, task):
        super().__init__(task)
        self.taglets = [FineTuneTaglet(task)]


class FineTuneTaglet(Taglet):
    def __init__(self, task):
        super().__init__(task)
        self.name = 'finetune'
        output_shape = self._get_model_output_shape(self.task.input_shape, self.model)
        self.model = torch.nn.Sequential(self.model,
                                         torch.nn.Linear(output_shape, len(self.task.classes)))
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
        for inputs in unlabeled_data_loader:
            if use_gpu:
                inputs = inputs.cuda()
            with torch.set_grad_enabled(False):
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                predicted_labels = predicted_labels + preds.detach().cpu().tolist()
        return predicted_labels

    def predict(self, data_loader, use_gpu):
        """
        predict on test data.
        :param data_loader: A dataloader containing images
        :param use_gpu: Whether or not to use the GPU
        :return: predictions
        """
    
        predicted_labels = []
        confidences = []
        self.model.eval()
        if use_gpu:
            self.model = self.model.cuda()
        else:
            self.model = self.model.cpu()
        for inputs in data_loader:
            if use_gpu:
                inputs = inputs.cuda()
        
            with torch.set_grad_enabled(False):
                for data in inputs:
                    data = torch.unsqueeze(data, dim=0)
                    outputs = self.model(data)
                    _, preds = torch.max(outputs, 1)
                    predicted_labels.append(preds.item())
                    confidences.append(torch.max(torch.nn.functional.softmax(outputs)).item())
    
        return predicted_labels, confidences
