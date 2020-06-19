from .module import Module
from ..pipeline import Taglet
import copy
import os
import logging
import torch
import torch.nn as nn

log = logging.getLogger(__name__)


class PrototypeModule(Module):
    def __init__(self, task):
        super().__init__(task)
        self.taglets = [PrototypeTaglet(task)]


class PrototypeTaglet(Taglet):
    def __init__(self, task, few_shot_support=1):
        super().__init__(task)
        self.name = 'prototype'
        self.few_shot_support = few_shot_support
        output_shape = self._get_model_output_shape(self.task.input_shape, self.model)
        self.classifier = Linear(output_shape, len(self.task.classes))
        self.save_dir = os.path.join('trained_models', self.name)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # Parameters needed to be updated based on freezing layer
        params_to_update = []
        for param in self.model.parameters():
            if param.requires_grad:
                params_to_update.append(param)
        for param in self.classifier.parameters():
            if param.requires_grad:
                params_to_update.append(param)
        self._params_to_update = params_to_update
        self.optimizer = torch.optim.Adam(self._params_to_update, lr=self.lr, weight_decay=1e-4)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

    @staticmethod
    def euclidean_metric(a, b):
        n = a.shape[0]
        m = b.shape[0]
        a = a.unsqueeze(1).expand(n, m, -1)
        b = b.unsqueeze(0).expand(n, m, -1)
        return torch.pow(a - b, 2).sum(2)

    def onn(self, i):
        dist = float("Inf")
        lab = ''
        for key, item in self.prototypes.items():
            prototype = item[0]
            rel_dist = PrototypeTaglet.euclidean_metric(prototype, i)
            if rel_dist < dist:
                dist = rel_dist
                lab = key
        return lab

    def train(self, train_data_loader, val_data_loader, use_gpu):
        """
        For 1-shot, use initial model
        """

        if use_gpu:
            self.model = self.model.cuda()
            self.classifier = self.classifier.cuda()
        else:
            self.model = self.model.cpu()
            self.classifier = self.classifier.cpu()

        best_model_to_save = None
        for epoch in range(self.num_epochs):
            log.info('epoch: {}'.format(epoch))

            # Train on training data
            train_loss = self._train_epoch(train_data_loader, use_gpu)
            log.info('train loss: {:.4f}'.format(train_loss))

            # Evaluation on validation data
            # Evaluation on validation data
            if not val_data_loader:
                val_loss = 0
                val_acc = 0
                continue
            val_loss, val_acc = self._validate_epoch(val_data_loader, use_gpu)
            log.info('validation loss: {:.4f}'.format(val_loss))
            log.info('validation acc: {:.4f}%'.format(val_acc*100))

            if self.lr_scheduler:
                self.lr_scheduler.step()

            if val_acc > self._best_val_acc:
                log.info("Deep copying new best model." +
                         "(validation of {:.4f}%, over {:.4f}%)".format(val_acc, self._best_val_acc))
                self._best_val_acc = val_acc
                best_model_to_save = copy.deepcopy(self.model.state_dict())
                torch.save(best_model_to_save, self.save_dir + '/model.pth.tar')

        log.info("Epoch {} result: ".format(epoch + 1))
        log.info("Average training loss: {:.4f}".format(train_loss))
        log.info("Average validation loss: {:.4f}".format(val_loss))
        log.info("Average validation accuracy: {:.4f}%".format(val_acc * 100))

        if self.select_on_val and best_model_to_save:
            # Reloads best model weights
            self.model.load_state_dict(best_model_to_save)

        self.model.eval()

        self.prototypes = {}
        for data in train_data_loader:
            with torch.set_grad_enabled(False):
                image, label = data[0], data[1]
                # Memorize
                if use_gpu:
                    image = image.cuda()
                    label = label.cuda()

                for img, lbl in zip(image, label):
                    proto = self.model(torch.unsqueeze(img, dim=0))
                    lbl = int(lbl.item())
                    try:
                        # 1-shot only no thoughts
                        self.prototypes[lbl].append(proto)
                    except:
                        self.prototypes[lbl] = [proto]

    def _train_epoch(self, train_data_loader, use_gpu):
        """
        Train for one epoch.
        :param train_data_loader: A dataloader containing training data
        :param cuda: Whether or not to use the GPU
        :return: None
        """
        self.model.train()
        self.classifier.train()
        running_loss = 0
        for batch_idx, batch in enumerate(train_data_loader):
            inputs = batch[0]
            labels = batch[1]
            if use_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()

            self.optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                representation = self.model(inputs)
                outputs = self.classifier(representation)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_data_loader.dataset)
        return epoch_loss

    def _validate_epoch(self, val_data_loader, use_gpu):
        """
        Validate for one epoch.
        :param val_data_loader: A dataloader containing validation data
        :return: None
        """
        self.model.eval()
        self.classifier.eval()
        running_loss = 0
        running_acc = 0
        for batch_idx, batch in enumerate(val_data_loader):

            inputs = batch[0]
            labels = batch[1]
            if use_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()
            with torch.set_grad_enabled(False):
                representation = self.model(inputs)
                outputs = self.classifier(representation)
                loss = self.criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

            running_loss += loss.item()
            running_acc += torch.sum(preds == labels)
        epoch_loss = running_loss / len(val_data_loader.dataset)
        epoch_acc = running_acc.item() / len(val_data_loader.dataset)
        return epoch_loss, epoch_acc

    def execute(self, unlabeled_data_loader, use_gpu):
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
                for data in inputs:
                    data = torch.unsqueeze(data, dim=0)
                    proto = self.model(data)
                    prediction = self.onn(proto)
                    predicted_labels.append(prediction)
        return predicted_labels


class Linear(nn.Module):
    def __init__(self, in_feature=64, out_feature=10):
        super().__init__()
        self.classifier = nn.Sequential(nn.Linear(in_feature, out_feature))

    def forward(self, x):
        x = self.classifier(x)
        return x


class ConvEncoder(nn.Module):
    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            self.conv_block(x_dim, hid_dim),
            self.conv_block(hid_dim, hid_dim),
            self.conv_block(hid_dim, hid_dim),
            self.conv_block(hid_dim, z_dim),
        )

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)

    def conv_block(self, in_channels, out_channels):
        bn = nn.BatchNorm2d(out_channels)
        nn.init.uniform_(bn.weight)  # For pytorch 1.2 or later
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            bn,
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
