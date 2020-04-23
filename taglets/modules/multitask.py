from taglets.data.custom_dataset import CustomDataset
from torch.utils import data

from .module import Module
from ..pipeline import Taglet
from ..scads.interface.scads import Scads

import os
import torch
import logging
import copy
import numpy as np
import torchvision.transforms as transforms
import torch.nn as nn

log = logging.getLogger(__name__)


class MultiTaskModel(nn.Module):
    def __init__(self, model, num_target, num_source):
        super().__init__()
        self.model = model
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, self.model.fc.in_features)
        self.fc_target = torch.nn.Linear(self.model.fc.in_features, num_target)
        self.fc_source = torch.nn.Linear(self.model.fc.in_features, num_source)

    def forward(self, x):
        x = self.model(x)
        return self.fc_target(x), self.fc_source(x)


class MultiTaskModule(Module):
    """
    A module that pre-trains on datasets selected from the SCADS and then
    transfers to available labeled data
    """
    def __init__(self, task):
        super().__init__(task)
        self.taglets = [MultiTaskTaglet(task)]


class MultiTaskTaglet(Taglet):
    def __init__(self, task):
        super().__init__(task)
        self.name = 'multitask'
        self.num_epochs = 5
        self.save_dir = os.path.join('trained_models', self.name)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def transform_image(self):
        """
        Get the transform to be used on an image.
        :return: A transform
        """
        data_mean = [0.485, 0.456, 0.406]
        data_std = [0.229, 0.224, 0.225]

        return transforms.Compose([
            transforms.Resize(self.task.input_shape),
            transforms.ToTensor(),
            transforms.Normalize(mean=data_mean, std=data_std)
        ])

    def _get_scads_data(self, num_batches, num_workers):
        root_path = Scads.get_root_path()
        Scads.open(self.task.scads_path)
        image_paths = []
        image_labels = []
        visited = set()
        for label, conceptnet_id in self.task.classes.items():
            target_node = Scads.get_node_by_conceptnet_id(conceptnet_id)
            neighbors = [edge.get_end_node() for edge in target_node.get_neighbors()]
            for neighbor in neighbors:
                if neighbor.get_conceptnet_id() in visited:
                    continue
                images = neighbor.get_images()
                images = [os.path.join(root_path, image) for image in images]
                if images:
                    image_paths.extend(images)
                    image_labels.extend([len(visited) for _ in range(len(images))])
                    visited.add(neighbor.get_conceptnet_id())
                    log.info("Source class found: {}".format(neighbor.get_conceptnet_id()))

        Scads.close()

        transform = self.transform_image()
        train_data = CustomDataset(image_paths,
                                   image_labels,
                                   transform)

        batch_size = min(len(train_data) // num_batches, 256)
        train_data_loader = torch.utils.data.DataLoader(train_data,
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        num_workers=num_workers)

        return train_data_loader, len(visited)

    def train(self, train_data_loader, val_data_loader, use_gpu):
        # Get Scadsdata and set up model
        batch_size, num_workers = train_data_loader.batch_size, train_data_loader.num_workers
        scads_train_data_loader, scads_num_classes = self._get_scads_data(len(train_data_loader) // batch_size,
                                                                          num_workers)
        log.info("Source classes found: {}".format(scads_num_classes))
        self.model = MultiTaskModel(self.model, len(self.task.classes), scads_num_classes)

        # Train
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
            train_loss, train_acc = self._multitask_train_epoch(train_data_loader,
                                                                scads_train_data_loader,
                                                                use_gpu)
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
                log.debug("Deep copying new best model." +
                          "(validation of {:.4f}%, over {:.4f}%)".format(
                              val_acc * 100, self._best_val_acc * 100))
                self._best_val_acc = val_acc
                best_model_to_save = copy.deepcopy(self.model.state_dict())
                if self.save_dir:
                    torch.save(best_model_to_save, self.save_dir + '/model.pth.tar')

        log.info("Epoch {} result: ".format(epoch + 1))
        log.info("Average training loss: {:.4f}".format(train_loss))
        log.info("Average training accuracy: {:.4f}%".format(train_acc * 100))
        log.info("Average validation loss: {:.4f}".format(val_loss))
        log.info("Average validation accuracy: {:.4f}%".format(val_acc * 100))

        val_dic = {'train': train_loss_list, 'validation': val_loss_list}
        if self.save_dir:
            self.save_plot('loss', val_dic, self.save_dir)
        val_dic = {'train': train_acc_list, 'validation': val_acc_list}
        if self.save_dir:
            self.save_plot('accuracy', val_dic, self.save_dir)

        if self.select_on_val and best_model_to_save:
            # Reloads best model weights
            self.model.load_state_dict(best_model_to_save)

    def _multitask_train_epoch(self, target_data_loader, scads_data_loader, use_gpu):
        """
        Train for one epoch.
        :param train_data_loader: A dataloader containing training data
        :param use_gpu: Whether or not to use the GPU
        :return: None
        """
        self.model.train()
        running_loss = 0
        running_acc = 0
        for batch_idx, (target_batch, source_batch) in enumerate(zip(target_data_loader, scads_data_loader)):
            target_inputs, target_labels = target_batch
            source_inputs, source_labels = source_batch
            if use_gpu:
                target_inputs = target_inputs.cuda()
                target_labels = target_labels.cuda()
                source_inputs = source_inputs.cuda()
                source_labels = source_labels.cuda()

            self.optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                target_outputs = self.model(target_inputs)[0]
                source_outputs = self.model(source_inputs)[1]
                target_loss = self.criterion(target_outputs, target_labels)
                source_loss = self.criterion(source_outputs, source_labels)
                total_loss = target_loss + source_loss
                total_loss.backward()
                self.optimizer.step()
                _, target_predictions = torch.max(target_outputs, 1)

            running_loss += target_loss.item()
            running_acc += torch.sum(target_predictions == target_labels)

        if not len(target_data_loader.dataset):
            return 0, 0

        epoch_loss = running_loss / len(target_data_loader.dataset)
        epoch_acc = running_acc.item() / len(target_data_loader.dataset)

        return epoch_loss, epoch_acc

    def _validate_epoch(self, val_data_loader, use_gpu):
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
            inputs = batch[0]
            labels = batch[1]
            if use_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()
            with torch.set_grad_enabled(False):
                outputs = self.model(inputs)[0]
                loss = self.criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

            running_loss += loss.item()
            running_acc += torch.sum(preds == labels)

        epoch_loss = running_loss / len(val_data_loader.dataset)
        epoch_acc = running_acc.item() / len(val_data_loader.dataset)

        return epoch_loss, epoch_acc

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
                outputs = self.model(inputs)[0]
                _, preds = torch.max(outputs, 1)
                predicted_labels = predicted_labels + preds.detach().cpu().tolist()
        return predicted_labels
