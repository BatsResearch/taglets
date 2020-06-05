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

log = logging.getLogger(__name__)


class TransferModule(Module):
    """
    A module that pre-trains on datasets selected from the SCADS and then
    transfers to available labeled data
    """
    def __init__(self, task):
        super().__init__(task)
        self.taglets = [TransferTaglet(task)]


class TransferTaglet(Taglet):
    def __init__(self, task):
        super().__init__(task)
        self.name = 'transfer'
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

    def _get_scads_data(self, batch_size, num_workers):
        root_path = Scads.get_root_path()
        Scads.open(self.task.scads_path)
        image_paths = []
        image_labels = []
        visited = set()
        for label, conceptnet_id in self.task.classes.items():
            target_node = Scads.get_node_by_conceptnet_id(conceptnet_id)
            neighbors = [edge.get_end_node() for edge in target_node.get_neighbors()]

            # Add target node
            if target_node not in visited:
                images = target_node.get_images()
                images = [os.path.join(root_path, image) for image in images]
                if images:
                    image_paths.extend(images)
                    image_labels.extend([len(visited) for _ in range(len(images))])
                    visited.add(target_node.get_conceptnet_id())
                    log.info("Source class found: {}".format(target_node.get_conceptnet_id()))

            # Add neighbors
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
        train_val_data = CustomDataset(image_paths,
                                       labels=image_labels,
                                       transform=transform)

        # 80% for training, 20% for validation
        train_percent = 0.8
        num_data = len(train_val_data)
        indices = list(range(num_data))
        train_split = int(np.floor(train_percent * num_data))
        np.random.shuffle(indices)
        train_idx = indices[:train_split]
        valid_idx = indices[train_split:]

        train_dataset = data.Subset(train_val_data, train_idx)
        val_dataset = data.Subset(train_val_data, valid_idx)

        train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=batch_size,
                                                        shuffle=False,
                                                        num_workers=num_workers)
        val_data_loader = torch.utils.data.DataLoader(val_dataset,
                                                      batch_size=batch_size,
                                                      shuffle=False,
                                                      num_workers=num_workers)

        return train_data_loader, val_data_loader, len(visited)

    def _set_num_classes(self, num_classes):
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)
        params_to_update = []
        for param in self.model.parameters():
            if param.requires_grad:
                params_to_update.append(param)
        self._params_to_update = params_to_update
        self.optimizer = torch.optim.Adam(self._params_to_update, lr=self.lr, weight_decay=1e-4)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

    def train(self, train_data_loader, val_data_loader, use_gpu):
        batch_size, num_workers = train_data_loader.batch_size, train_data_loader.num_workers
        scads_train_data_loader, scads_val_data_loader, scads_num_classes = self._get_scads_data(batch_size,
                                                                                                 num_workers)
        log.info("Source classes found: {}".format(scads_num_classes))
        self._set_num_classes(scads_num_classes)
        self._train(scads_train_data_loader, scads_val_data_loader, use_gpu)

        # TODO: Freeze layers
        self._set_num_classes(len(self.task.classes))
        self._train(train_data_loader, val_data_loader, use_gpu)

    def _train(self, train_data_loader, val_data_loader, use_gpu):
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
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                predicted_labels = predicted_labels + preds.detach().cpu().tolist()
        return predicted_labels
