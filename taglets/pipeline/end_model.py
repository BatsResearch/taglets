import torch
import os
import torch
import pandas as pd
from PIL import Image
from ..pipeline import Trainable


class EndModel(Trainable):
    def __init__(self, task):
        super().__init__(task)
        self.name = 'end model'
        self.save_dir = os.path.join('trained_models', task.phase ,str(task.task_id), self.name)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    @staticmethod
    def soft_cross_entropy(prediction, target):
        prediction = prediction.double()
        target = target.double()
        logs = torch.nn.LogSoftmax(dim=1)
        return torch.mean(torch.sum(-target * logs(prediction), 1))

    def _train_epoch(self, train_data_loader, use_gpu, testing):
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
            if testing:
                if batch_idx >= 1:
                    break
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
            running_acc += torch.sum(torch.max(outputs, 1)[1] == torch.max(labels, 1)[1]).item() #/ float(len(labels))

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
        test_imgs = []
        self.model.eval()

        for inputs, _ in data_loader:
            if use_gpu:
                inputs = inputs.cuda()

            with torch.set_grad_enabled(False):
                for data in inputs:
                    data = torch.unsqueeze(data, dim=0)
                    outputs = self.model(data)
                    _, preds = torch.max(outputs, 1)
                    predicted_labels.append(str(preds.item()))
                    confidences.append(torch.max(torch.nn.functional.softmax(outputs)).item())

        assert len(predicted_labels) == len(test_imgs)

        pred_df = pd.DataFrame({'id': test_imgs, 'class': predicted_labels})
        prbs_df = pd.DataFrame({'id': test_imgs, 'confidence': confidences})
        
        return pred_df.to_dict(), prbs_df.to_dict()
