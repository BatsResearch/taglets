import numpy as np
import torch
import torchvision.models as models
import random
import copy
import torchvision.models as models
import os
import torch
from models import custom_models


class Taglet:
    """
    A class that produces votes for unlabeled images.
    """
    def __init__(self, task):
        """
        Create a new Taglet.
        :param task: The current task
        """
        self.name = 'base'
        self.task = task
        self.lr = 0.0001
        self.criterion = torch.nn.CrossEntropyLoss()
        self.seed = 0
        self.num_epochs = 1
        self.select_on_val = True   # If true, save model on the best validation performance
        self.pretrained = True      # If true, we can load from pretrained model

        if task.number_of_channels == 1:
            self.model = custom_models.MnistResNet()
            self.model.pretrained = self.pretrained
        else:
            self.model = models.resnet18(pretrained=self.pretrained)

        self.model.fc = torch.nn.Linear(self.model.fc.in_features, len(self.task.classes))

        self._init_random(self.seed)
        # Parameters needed to be updated based on freezing layer
        params_to_update = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)
        self._params_to_update = params_to_update
        self.optimizer = torch.optim.Adam(self._params_to_update, lr=self.lr, weight_decay=1e-4)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        self.save_dir = os.path.join('trained_models', str(task.task_id), self.name)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

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

    def _train_epoch(self, train_data_loader, use_gpu):
        """
        Train for one epoch.
        :param train_data_loader: A dataloader containing training data
        :param cuda: Whether or not to use the GPU
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
            running_acc += torch.sum(preds == labels.data)

            if batch_idx >= 1:
                break

        epoch_loss = running_loss / len(train_data_loader.dataset)
        epoch_acc = running_acc / len(train_data_loader.dataset)

        return epoch_loss, epoch_acc

    def _validate_epoch(self, val_data_loader, use_gpu):
        """
        Validate for one epoch.
        :param val_data_loader: A dataloader containing validation data
        :return: None
        """
        self.model.eval()
        running_loss = 0
        running_acc = 0
        for batch_idx, batch in enumerate(val_data_loader):
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
            running_acc += torch.sum(preds == labels.data)
            if batch_idx >= 2:
                break

        epoch_loss = running_loss / len(val_data_loader.dataset)
        epoch_acc = running_acc / len(val_data_loader.dataset)

        return epoch_loss, epoch_acc

    def train(self, train_data_loader, val_data_loader, test_data_loader, use_gpu):
        """
        Train the Taglet.
        :param train_data_loader: A dataloader containing training data
        :param val_data_loader: A dataloader containing validation data
        :param test_data_loader: A dataloader containing test dat
        :param use_gpu: Whether or not to use the GPU
        :return:
        """
        if use_gpu:
            self.model = self.model.cuda()
        else:
            self.model = self.model.cpu()

        best_model_to_save = None
        for epoch in range(self.num_epochs):
            self.log('epoch: {}'.format(epoch))

            # Train on training data
            train_loss, train_acc = self._train_epoch(train_data_loader, use_gpu)
            self.log('train loss: {:.4f}'.format(train_loss))
            self.log('train acc: {:.4f}%'.format(train_acc*100))

            # Evaluation on validation data
            val_loss, val_acc = self._validate_epoch(val_data_loader, use_gpu)
            self.log('validation loss: {:.4f}'.format(val_loss))
            self.log('validation acc: {:.4f}%'.format(val_acc*100))

            if self.lr_scheduler:
                self.lr_scheduler.step()

            if val_acc > self._best_val_acc:
                self.log("Deep copying new best model." +
                         "(validation of {:.4f}%, over {:.4f}%)".format(val_acc, self._best_val_acc))
                self._best_val_acc = val_acc
                best_model_to_save = copy.deepcopy(self.model.state_dict())
                torch.save(best_model_to_save, self.save_dir + 'model.pth.tar')

        self.log("Epoch {} result: ".format(epoch + 1))
        self.log("Average training loss: {:.4f}".format(train_loss))
        self.log("Average training accuracy: {:.4f}%".format(train_acc * 100))
        self.log("Average validation loss: {:.4f}".format(val_loss))
        self.log("Average validation accuracy: {:.4f}%".format(val_acc * 100))

        if self.select_on_val and best_model_to_save:
            # Reloads best model weights
            self.model.load_state_dict(best_model_to_save)

        test_loss, test_acc = self._validate_epoch(test_data_loader, use_gpu)
        self.log('test loss: {:.4f}'.format(test_loss))
        self.log('test acc: {:.4f}%'.format(test_acc * 100))

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
                inputs = inputs[0].cuda()
            else:
                inputs = inputs[0].cpu()
            with torch.set_grad_enabled(False):
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                predicted_labels.extend(preds)
        return predicted_labels


class FineTuneTaglet(Taglet):
    def __init__(self, task):
        super().__init__(task)


class TransferTaglet(Taglet):
    def __init__(self, task):
        super().__init__(task)
        self.model = models.resnet18(pretrained=self.pretrained)
        self.lr = 0.001
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.num_epochs = 50
        self.use_gpu = True
        self.batch_size = 32

    def pretrain(self, images, labels):
        raise NotImplementedError

    def finetune(self, images, labels):
        num_images = images.shape[0]

        # Top: not sure if this is the most efficient way of doing it
        self.model = self.model.train()
        if self.use_gpu:
            self.model = self.model.cuda()
        else:
            self.model = self.model.cpu()

        for epoch in range(self.num_epochs):
            perm = torch.randperm(num_images)
            for i in range(0, num_images, self.batch_size):
                self.optimizer.zero_grad()

                ind = perm[i: i + self.batch_size]
                batch_images = images[ind]
                batch_labels = labels[ind]

                if self.use_gpu:
                    batch_images = batch_images.cuda()
                    batch_labels = batch_labels.cuda()

                logits = self.model(batch_images)
                loss = self.criterion(logits, batch_labels)

                loss.backward()
                self.optimizer.step()


class MTLTaglet(Taglet):
    def __init__(self, task):
        super().__init__(task)

    def train(self):
        raise NotImplementedError

    def finetune(self):
        raise NotImplementedError


class PrototypeTaglet(Taglet):
    def __init__(self, task):
        super().__init__(task)
        # Self.model = Peilin will take care of this
