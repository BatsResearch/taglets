import torch
import os
import torch
import pandas as pd
from PIL import Image
from taglets.taglet import Trainable


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
            running_acc += torch.sum(torch.max(outputs, 1)[1] == torch.max(labels, 1)[1]).item() #/ float(len(labels))


            # if batch_idx >= 1:
            #     break

        epoch_loss = running_loss / len(train_data_loader.dataset)
        epoch_acc = running_acc / len(train_data_loader.dataset)

        return epoch_loss, epoch_acc

    def predict(self, evaluation_image_path, number_of_channels, transform, use_gpu):
        """
        predict on test data.
        :param evaluation_image_path: path to the evaluation data
        :param use_gpu: Whether or not to use the GPU
        :return: predictions
        """

        predictons = []
        test_imgs = []
        self.model.eval()

        for image in os.listdir(evaluation_image_path):
            test_imgs.append(image)
            img = os.path.join(evaluation_image_path, image)
            img = Image.open(img)
            if number_of_channels == 3:
                img = img.convert('RGB')

            if transform is not None:
                img = transform(img)

            if use_gpu:
                img = img.cuda()

            with torch.set_grad_enabled(False):
                img = torch.unsqueeze(img, dim=0)
                outputs = self.model(img)
                _, preds = torch.max(outputs, 1)
                predictons.append(preds.item())

        assert len(predictons) == len(test_imgs)
        df = pd.DataFrame({'id': test_imgs, 'label': predictons})

        return df.to_dict()
