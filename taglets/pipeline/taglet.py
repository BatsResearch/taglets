from ..models import MnistResNet

import copy
import logging
import matplotlib.pyplot as plt
import numpy as np
import random
import torchvision.models as models
import torch


log = logging.getLogger(__name__)


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

    def train(self, train_data_loader, val_data_loader, use_gpu):
        """
        Train the Trainable.
        :param train_data_loader: A dataloader containing training data
        :param val_data_loader: A dataloader containing validation data
        :param use_gpu: Whether or not to use the GPU
        :return:
        """
        log.info('Beginning training')

        if use_gpu:
            self.model = self.model.cuda()
        else:
            self.model = self.model.cpu()

        best_model_to_save = None
        train_loss_list = []
        train_acc_list = []
        val_loss_list = []
        val_acc_list = []

        for epoch in range(self.num_epochs):
            log.info('Epoch: %s', epoch)

            # Train on training data
            train_loss, train_acc = self._train_epoch(train_data_loader, use_gpu)
            train_loss_list.append(train_loss)
            train_acc_list.append(train_acc)
            log.info('train loss: {:.4f}'.format(train_loss))
            log.info('train acc: {:.4f}%'.format(train_acc*100))

            # Evaluation on validation data
            if not val_data_loader:
                val_loss = 0
                val_acc = 0
                continue
            val_loss, val_acc = self._validate_epoch(val_data_loader, use_gpu)
            val_loss_list.append(val_loss)
            val_acc_list.append(val_acc)
            log.info('validation loss: {:.4f}'.format(val_loss))
            log.info('validation acc: {:.4f}%'.format(val_acc*100))

            if self.lr_scheduler:
                self.lr_scheduler.step()

            if val_acc > self._best_val_acc:
                log.info("Deep copying new best model." +
                         "(validation of {:.4f}%, over {:.4f}%)".format(val_acc, self._best_val_acc))
                self._best_val_acc = val_acc
                best_model_to_save = copy.deepcopy(self.model.state_dict())
                if self.save_dir:
                    torch.save(best_model_to_save, self.save_dir + '/model.pth.tar')

        log.info("Epoch {} result: ".format(epoch + 1))
        log.info("Average training loss: {:.4f}".format(train_loss))
        log.info("Average training accuracy: {:.4f}%".format(train_acc * 100))
        log.info("Average validation loss: {:.4f}".format(val_loss))
        log.info("Average validation accuracy: {:.4f}%".format(val_acc * 100))


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
    A trainable model that produces votes for unlabeled images
    """
    def execute(self, unlabeled_data_loader, use_gpu):
        """
        Execute the Taglet on unlabeled images.
        :param unlabeled_data_loader: A dataloader containing unlabeled data
        :param use_gpu: Whether or not the use the GPU
        :return: A list of predicted labels
        """
        raise NotImplementedError



