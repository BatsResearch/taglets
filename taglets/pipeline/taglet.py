import numpy as np
import random
import copy
import torchvision.models as models
import os
import torch
from ..models import MnistResNet, Linear, ConvEncoder
import matplotlib.pyplot as plt


class Trainable:
    """
    A class with a trainable model.
    """
    def __init__(self, task):
        """
        Create a new Trainable.
        :param task: The current task
        """
        self.name = 'base'
        self.task = task
        self.lr = 0.001
        self.criterion = torch.nn.CrossEntropyLoss()
        self.seed = 0
        self.num_epochs = 10
        self.select_on_val = True   # If true, save model on the best validation performance
        self.pretrained = task.pretrained      # If true, we can load from pretrained model
        self.save_dir = None

        if task.number_of_channels == 1:
            self.model = MnistResNet()
        else:
            self.model = models.resnet18(pretrained=self.pretrained)

        self.model.fc = torch.nn.Linear(self.model.fc.in_features, len(self.task.classes))

        self.model.pretrained = self.pretrained

        self._init_random(self.seed)
        # Parameters needed to be updated based on freezing layer
        params_to_update = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)
        self._params_to_update = params_to_update
        self.optimizer = torch.optim.Adam(self._params_to_update, lr=self.lr, weight_decay=1e-4)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

        self.log = print
        self._best_val_acc = 0.0

    @staticmethod
    def _init_random(seed):
        """
        Initialize random numbers with a seed.
        :param seed: The seed to initialize with
        :return: None
        """
        torch.backends.cudnn.deterministic = True

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

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
                loss = self.criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                loss.backward()
                self.optimizer.step()

            running_loss += loss.item()
            running_acc += torch.sum(preds == labels)

        if not len(train_data_loader.dataset):
            return 0, 0

        epoch_loss = running_loss / len(train_data_loader.dataset)
        epoch_acc = running_acc.item() / len(train_data_loader.dataset)

        return epoch_loss, epoch_acc

    def _validate_epoch(self, val_data_loader, use_gpu, testing):
        """
        Validate for one epoch.
        :param val_data_loader: A dataloader containing validation data
        :param use_gpu: Whether or not to use the GPU
        :return: None
        """
        self.model.eval()
        running_loss = 0
        running_acc = 0
        for batch_idx, batch in enumerate(val_data_loader):
            if testing:
                if batch_idx >= 2:
                    break
            inputs = batch[0]
            labels = batch[1]
            if use_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()
            with torch.set_grad_enabled(False):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

            running_loss += loss.item()
            running_acc += torch.sum(preds == labels)

        epoch_loss = running_loss / len(val_data_loader.dataset)
        epoch_acc = running_acc.item() / len(val_data_loader.dataset)

        return epoch_loss, epoch_acc

    def train(self, train_data_loader, val_data_loader, use_gpu, testing):
        """
        Train the Trainable.
        :param train_data_loader: A dataloader containing training data
        :param val_data_loader: A dataloader containing validation data
        :param use_gpu: Whether or not to use the GPU
        :return:
        """
        self.log('...........'+self.name+'...........')

        if use_gpu:
            self.model = self.model.cuda()
        else:
            self.model = self.model.cpu()

        best_model_to_save = None
        train_loss_list = []
        train_acc_list = []
        val_loss_list = []
        val_acc_list = []

        if testing:
            num_epochs = 1
        else:
            num_epochs = self.num_epochs
        for epoch in range(num_epochs):
            self.log('epoch: {}'.format(epoch))

            # Train on training data
            train_loss, train_acc = self._train_epoch(train_data_loader, use_gpu, testing)
            train_loss_list.append(train_loss)
            train_acc_list.append(train_acc)
            self.log('train loss: {:.4f}'.format(train_loss))
            self.log('train acc: {:.4f}%'.format(train_acc*100))

            # Evaluation on validation data
            if not val_data_loader:
                val_loss = 0
                val_acc = 0
                continue
            val_loss, val_acc = self._validate_epoch(val_data_loader, use_gpu, testing)
            val_loss_list.append(val_loss)
            val_acc_list.append(val_acc)
            self.log('validation loss: {:.4f}'.format(val_loss))
            self.log('validation acc: {:.4f}%'.format(val_acc*100))

            if self.lr_scheduler:
                self.lr_scheduler.step()

            if val_acc > self._best_val_acc:
                self.log("Deep copying new best model." +
                         "(validation of {:.4f}%, over {:.4f}%)".format(val_acc, self._best_val_acc))
                self._best_val_acc = val_acc
                best_model_to_save = copy.deepcopy(self.model.state_dict())
                if self.save_dir:
                    torch.save(best_model_to_save, self.save_dir + '/model.pth.tar')

        self.log("Epoch {} result: ".format(epoch + 1))
        self.log("Average training loss: {:.4f}".format(train_loss))
        self.log("Average training accuracy: {:.4f}%".format(train_acc * 100))
        self.log("Average validation loss: {:.4f}".format(val_loss))
        self.log("Average validation accuracy: {:.4f}%".format(val_acc * 100))


        val_dic = {'train':train_loss_list, 'validation':val_loss_list}
        if self.save_dir:
            self.save_plot('loss',val_dic,self.save_dir)
        val_dic = {'train':train_acc_list, 'validation':val_acc_list}
        if self.save_dir:
            self.save_plot('accuracy',val_dic,self.save_dir)

        if self.select_on_val and best_model_to_save:
            # Reloads best model weights
            self.model.load_state_dict(best_model_to_save)

    def save_plot(self, plt_mode, val_dic, save_dir):
        plt.figure()
        colors = ['r', 'b', 'g']

        counter = 0
        for k, v in val_dic.items():
            val = [np.round(float(i), decimals=3) for i in v]
            plt.plot(val, color=colors[counter], label=k + ' ' + plt_mode)
            counter += 1

        if plt_mode == 'loss':
            plt.legend(loc='upper right')
        elif plt_mode == 'accuracy':
            plt.legend(loc='lower right')
        title = '_vs.'.join(list(val_dic.keys()))
        plt.title(title + ' ' + plt_mode)
        plt.savefig(save_dir + '/' + plt_mode + '_' + title + '.pdf')
        plt.close()


class Taglet(Trainable):
    """
    A class that produces votes for unlabeled images.
    """
    def execute(self, unlabeled_data_loader, use_gpu, testing):
        """
        Execute the Taglet on unlabeled images.
        :param unlabeled_data_loader: A dataloader containing unlabeled data
        :param use_gpu: Whether or not the use the GPU
        :return: A list of predicted labels
        """
        raise NotImplementedError


class FineTuneTaglet(Taglet):
    def __init__(self, task):
        super().__init__(task)
        self.name = 'finetune'
        self.num_epochs = 5
        to_load_dir = os.path.join('trained_models', "base", str(task.task_id), self.name)
        self.save_dir = os.path.join('trained_models', task.phase, str(task.task_id), self.name)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        if task.phase == 'adaptation': # in adaptation model, load from model trained on base phase
            self.model.load_state_dict(torch.load(to_load_dir + '/model.pth.tar'))
            self.log('^^^^^^^^^ adaptation: loading from base')

    def execute(self, unlabeled_data_loader, use_gpu, testing):
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
            if testing:
                break
        return predicted_labels


class PrototypeTaglet(Taglet):
    def __init__(self, task, few_shot_support=1):
        super().__init__(task)
        self.name = 'prototype'
        self.few_shot_support = few_shot_support
        self.model = ConvEncoder()
        self.classifier = Linear()
        self.num_epochs = 3
        to_load_dir = os.path.join('trained_models', "base", str(task.task_id), self.name)
        self.save_dir = os.path.join('trained_models', task.phase, str(task.task_id), self.name)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        if task.phase == 'adaptation': # in adaptation model, load from model trained on base phase
            self.model.load_state_dict(torch.load(to_load_dir + '/model.pth.tar'))
            self.log('^^^^^^^^^ adaptation: loading from base')

        # Parameters needed to be updated based on freezing layer
        params_to_update = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)

        for name, param in self.classifier.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)
        self._params_to_update = params_to_update

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

    def train(self, train_data_loader, val_data_loader, use_gpu, testing):
        """
        For 1-shot, use pretrained model
        """
        self.log('...........'+self.name+'...........')

        if use_gpu:
            self.model = self.model.cuda()
            self.classifier = self.classifier.cuda()
        else:
            self.model = self.model.cpu()
            self.classifier = self.classifier.cpu()

        best_model_to_save = None
        if testing:
            num_epochs = 1
        else:
            num_epochs = self.num_epochs
        for epoch in range(num_epochs):
            self.log('epoch: {}'.format(epoch))

            # Train on training data
            train_loss = self._train_epoch(train_data_loader, use_gpu, testing)
            self.log('train loss: {:.4f}'.format(train_loss))

            # Evaluation on validation data
            val_loss, val_acc = self._validate_epoch(val_data_loader, use_gpu, testing)
            self.log('validation loss: {:.4f}'.format(val_loss))
            self.log('validation acc: {:.4f}%'.format(val_acc*100))

            if self.lr_scheduler:
                self.lr_scheduler.step()

            if val_acc > self._best_val_acc:
                self.log("Deep copying new best model." +
                         "(validation of {:.4f}%, over {:.4f}%)".format(val_acc, self._best_val_acc))
                self._best_val_acc = val_acc
                best_model_to_save = copy.deepcopy(self.model.state_dict())
                torch.save(best_model_to_save, self.save_dir + '/model.pth.tar')

        self.log("Epoch {} result: ".format(epoch + 1))
        self.log("Average training loss: {:.4f}".format(train_loss))
        self.log("Average validation loss: {:.4f}".format(val_loss))
        self.log("Average validation accuracy: {:.4f}%".format(val_acc * 100))

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
                    if testing:
                        break

    def _train_epoch(self, train_data_loader, use_gpu, testing):
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
                representation = self.model(inputs)
                outputs = self.classifier(representation)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_data_loader.dataset)
        return epoch_loss

    def _validate_epoch(self, val_data_loader, use_gpu, testing):
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
            if testing:
                if batch_idx >= 2:
                    break

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

    def execute(self, unlabeled_data_loader, use_gpu, testing):
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
                for data, ix in zip(inputs,index):
                    data = torch.unsqueeze(data, dim=0)
                    proto = self.model(data)
                    prediction = self.onn(proto)
                    predicted_labels.append(prediction)
            if testing:
                break
        return predicted_labels


class TransferTaglet(Taglet):
    def __init__(self, task):
        super().__init__(task)
        self.name = 'transfer'
        self.num_epochs = 5
        to_load_dir = os.path.join('trained_models', "base", str(task.task_id), self.name)
        self.save_dir = os.path.join('trained_models', task.phase, str(task.task_id), self.name)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        if task.phase == 'adaptation':  # in adaptation model, load from model trained on base phase
            self.model.load_state_dict(torch.load(to_load_dir + '/model.pth.tar'))
            self.log('^^^^^^^^^ adaptation: loading from base')

    def execute(self, unlabeled_data_loader, use_gpu, testing):
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
            if testing:
                break
        return predicted_labels
