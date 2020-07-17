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

        # Parameters needed to be updated based on freezing layer
        params_to_update = []
        for param in self.model.parameters():
            if param.requires_grad:
                params_to_update.append(param)
        self._params_to_update = params_to_update
        self.optimizer = torch.optim.Adam(self._params_to_update, lr=self.lr, weight_decay=1e-4)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

    @staticmethod
    def soft_cross_entropy(prediction, target):
        prediction = prediction.double()
        target = target.double()
        logs = torch.nn.LogSoftmax(dim=1)
        return torch.mean(torch.sum(-target * logs(prediction), 1))

    def _train_epoch(self, train_data_loader, use_gpu):
        """
        Train for one epoch.
        :param train_data_loader: A dataloader containing training data
        :param use_gpu: Whether or not to use the GPU
        :return: None
        """
        self.model.train()
        running_loss = 0
        running_acc = 0
        for batch_idx, batch in enumerate(train_data_loader):
            inputs = batch[0]
            labels = batch[1]
            if use_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()

            self.optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = self.model(inputs)
                loss = EndModel.soft_cross_entropy(outputs, labels)
                loss.backward()
                self.optimizer.step()

            running_loss += loss.item()
            running_acc += torch.sum(torch.max(outputs, 1)[1] == torch.max(labels, 1)[1]).item()

        epoch_loss = running_loss / len(train_data_loader.dataset)
        epoch_acc = running_acc / len(train_data_loader.dataset)

        return epoch_loss, epoch_acc

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
