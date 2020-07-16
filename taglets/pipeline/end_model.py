from .taglet import Trainable

import os
import torch


class EndModel(Trainable):
    def __init__(self, task):
        super().__init__(task)
        self.name = 'end model'
        output_shape = self._get_model_output_shape(self.task.input_shape, self.model)
        self.model = torch.nn.Sequential(self.model,
                                         torch.nn.Linear(output_shape, len(self.task.classes)))
        self.save_dir = os.path.join('trained_models', self.name)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.criterion = self.soft_cross_entropy

    @staticmethod
    def soft_cross_entropy(prediction, target):
        prediction = prediction.double()
        target = target.double()
        logs = torch.nn.LogSoftmax(dim=1)
        return torch.mean(torch.sum(-target * logs(prediction), 1))

    @staticmethod
    def _get_train_acc(outputs, labels):
        return torch.sum(torch.max(outputs, 1)[1] == torch.max(labels, 1)[1])

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
        for inputs in data_loader:
            if use_gpu:
                inputs = inputs.cuda()

            with torch.set_grad_enabled(False):
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                predicted_labels.append(preds.item())
                confidences.append(torch.max(torch.nn.functional.softmax(outputs)).item())
        
        return predicted_labels, confidences

    def evaluate(self, data_loader, use_gpu):
        """
        Evaluate on labeled data.

        :param data_loader: A dataloader containing images and ground truth labels
        :param use_gpu: Whether or not to use the GPU
        :return: accuracy
        """

        self.model.eval()
        correct = 0
        total = 0

        for batch in data_loader:
            inputs = batch[0]
            labels = batch[1]
            if use_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()

            with torch.set_grad_enabled(False):
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                for pred, label in zip(preds, labels):
                    total += 1
                    if pred == label:
                        correct += 1

        return correct / total
