import numpy as np
import torch
import torchvision.models as models
from numpy import inf
from abc import abstractmethod
import random
import copy
import torchvision.models as models
import os
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch
from custom_dataset import CustomDataSet
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils import data
from models import custom_models


class Taglet:
    """
    Taglet class
    """

    def __init__(self, task):
        """Initialize based on configuration dictionary"""

        self.name = 'base'
        self.task = task
        self.lr = 0.0001
        self.criterion = torch.nn.CrossEntropyLoss()
        self.seed = 0
        self.num_epochs = 1
        self.select_on_val = True   # If true, save model on the best validation performance
        self.batch_size = 32
        self.pretrained = True      # If true, we can load from pretrained model
        self.num_workers = 2

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
        self.device = self._init_device()
        self.model = self.model.to(self.device)
        self.model.eval()

        self.train_data_loader, self.val_data_loader, self.test_data_loader = self.load_data()

    @staticmethod
    def _init_device():
        """
        setup GPU device if available
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return device

    @staticmethod
    def _init_random(seed):
        torch.backends.cudnn.deterministic = True

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def _train_epoch(self):
        """
        Training for an epoch
        """
        self.model.train()
        running_loss = 0
        running_acc = 0
        for batch_idx, batch in enumerate(self.train_data_loader):
            inputs = batch[0].to(self.device)
            labels = batch[1].to(self.device)

            self.optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                loss.backward()
                self.optimizer.step()

            running_loss += loss.item()
            running_acc += torch.sum(preds == labels.data)

            if batch_idx >=1:
                break

        epoch_loss = running_loss / len(self.train_data_loader.dataset)
        epoch_acc = running_acc / len(self.train_data_loader.dataset)

        return epoch_loss, epoch_acc

    def _validate_epoch(self, data_loader):
        """ validating/testing """

        self.model.eval()
        running_loss = 0
        running_acc = 0
        for batch_idx, batch in enumerate(data_loader):
            inputs = batch[0].to(self.device)
            labels = batch[1].to(self.device)
            with torch.set_grad_enabled(False):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

            running_loss += loss.item()
            running_acc += torch.sum(preds == labels.data)
            if batch_idx >= 2:
                break

        epoch_loss = running_loss / len(data_loader.dataset)
        epoch_acc = running_acc / len(data_loader.dataset)

        return epoch_loss, epoch_acc

    def train(self):
        """
        Training phase
        """
        best_model_to_save = None
        for epoch in range(self.num_epochs):
            self.log('epoch: {}'.format(epoch))

            # Train on training data
            train_loss, train_acc = self._train_epoch()
            self.log('train loss: {:.4f}'.format(train_loss))
            self.log('train acc: {:.4f}%'.format(train_acc*100))

            # Evaluation on validation data
            val_loss, val_acc = self._validate_epoch(self.val_data_loader)
            self.log('validation loss: {:.4f}'.format(val_loss))
            self.log('validation acc: {:.4f}%'.format(val_acc*100))

            if self.lr_scheduler:
                self.lr_scheduler.step()

            if val_acc > self._best_val_acc:
                self.log("Deep copying new best model. (validation of {:.4f}%, over {:.4f}%)".format(val_acc, self._best_val_acc))
                self._best_val_acc = val_acc
                best_model_to_save = copy.deepcopy(self.model.state_dict())
                torch.save(best_model_to_save, self.save_dir + 'model.pth.tar')

        self.log("Epoch {} result: ".format(epoch + 1))
        self.log("Average training loss: {:.4f}".format(train_loss))
        self.log("Average training accuracy: {:.4f}%".format(train_acc * 100))
        self.log("Average validation loss: {:.4f}".format(val_loss))
        self.log("Average validation accuracy: {:.4f}%".format(val_acc * 100))

        if self.select_on_val:
            # Reloads best model weights
            self.model.load_state_dict(best_model_to_save)

        test_loss, test_acc = self._validate_epoch(self.test_data_loader)
        self.log('test loss: {:.4f}'.format(test_loss))
        self.log('test acc: {:.4f}%'.format(test_acc * 100))

    def execute(self, unlabeled_images, batch_size=64, use_gpu=True):
        """
        Execute the taglet on all unlabeled images.
        :return: A batch of labels
        """
        num_images = unlabeled_images.shape[0]
        self.model = self.model.eval()
        if use_gpu:
            self.model = self.model.cuda()
        else:
            self.model = self.model.cpu()
        list_logits = []
        for i in range(0, num_images, batch_size):
            batch_images = unlabeled_images[i: i + batch_size]

            if use_gpu:
                batch_images = batch_images.cuda()

            logits = self.model(batch_images)
            list_logits.append(logits.cpu().detach().numpy())
        all_logits = np.concatenate(list_logits)
        predicted_labels = all_logits.argmax(axis=1)
        return predicted_labels

    def load_data(self):

        if self.task.number_of_channels == 3:
            data_mean = [0.485, 0.456, 0.406]
            data_std = [0.229, 0.224, 0.225]
        elif self.task.number_of_channels == 1:
            data_mean = [0.5]
            data_std = [0.5]

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=data_mean, std=data_std)
        ])

        image_names = [img_name for img_name, label in self.task.labeled_images]
        image_labels = [label for img_name, label in self.task.labeled_images]
        train_val_test_data = CustomDataSet(self.task.unlabeled_image_path,
                                            image_names,
                                            image_labels,
                                            transform,
                                            self.task.number_of_channels)

        # 70% for training, 15% for validation, and 15% for test
        train_percent = 0.7
        num_data = len(train_val_test_data)
        indices = list(range(num_data))
        train_split = int(np.floor(train_percent * num_data))
        np.random.shuffle(indices)
        train_idx = indices[:train_split]
        val_test_data = indices[train_split:]
        val_split = int(np.floor(len(val_test_data)/2))
        valid_idx = val_test_data[:val_split]
        test_idx = val_test_data[val_split:]

        train_set = data.Subset(train_val_test_data, train_idx)
        val_set = data.Subset(train_val_test_data, valid_idx)
        test_set = data.Subset(train_val_test_data, test_idx)

        # test_data = datasets.ImageFolder(self.task.test_image_path, transform= transform)

        train_data_loader = torch.utils.data.DataLoader(train_set,
                                                        batch_size=self.batch_size,
                                                        shuffle=False,
                                                        num_workers=self.num_workers)
        val_data_loader = torch.utils.data.DataLoader(val_set,
                                                      batch_size=self.batch_size,
                                                      shuffle=False,
                                                      num_workers=self.num_workers)
        test_data_loader = torch.utils.data.DataLoader(test_set,
                                                       batch_size=self.batch_size,
                                                       shuffle=False,
                                                        num_workers=self.num_workers)

        self.log('number of training data: %d' % len(train_data_loader.dataset))
        self.log('number of validation data: %d' % len(val_data_loader.dataset))
        self.log('number of test data: %d' % len(test_data_loader.dataset))

        return train_data_loader, val_data_loader, test_data_loader


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
