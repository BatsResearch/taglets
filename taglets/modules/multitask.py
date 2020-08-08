from .module import Module
from ..data.custom_dataset import CustomDataset
from ..pipeline import Taglet
from ..scads.interface.scads import Scads

import os
import torch
import logging
import numpy as np
import torchvision.transforms as transforms
import torch.nn as nn

log = logging.getLogger(__name__)


class MultiTaskModel(nn.Module):
    def __init__(self, model, num_target, num_source, input_shape):
        super().__init__()
        self.model = model
        self.num_target= num_target
        self.num_source = num_source
        self.base = nn.Sequential(*list(self.model.children())[:-1])
        output_shape = self._get_model_output_shape(input_shape, self.model)
        self.fc_target = torch.nn.Linear(output_shape, self.num_target)
        self.fc_source = torch.nn.Linear(output_shape, self.num_source)

    def forward(self, x):
        x = self.base(x)
        x = torch.flatten(x, 1)
        return self.fc_target(x)

    def forward_source(self, x):
        x = self.base(x)
        x = torch.flatten(x, 1)
        return self.fc_source(x)

    def _get_model_output_shape(self, in_size, mod):
        """
        Adopt from https://gist.github.com/lebedov/0db63ffcd0947c2ea008c4a50be31032
        Compute output size of Module `mod` given an input with size `in_size`
        :param in_size: input shape (height, width)
        :param mod: PyTorch model
        :return:
        """
        mod = mod.cpu()
        f = mod(torch.rand(2, 3, *in_size))
        return int(np.prod(f.size()[1:]))


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
            transforms.RandomCrop(self.task.input_shape),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=data_mean, std=data_std)
        ])

    def _get_scads_data(self):
        root_path = Scads.get_root_path()
        Scads.open(self.task.scads_path)
        image_paths = []
        image_labels = []
        visited = set()
        for conceptnet_id in self.task.classes:
            target_node = Scads.get_node_by_conceptnet_id(conceptnet_id)

            neighbors = [edge.get_end_node() for edge in target_node.get_neighbors()]
            # Add target node
            if target_node.get_conceptnet_id() not in visited:
                images = target_node.get_images_whitelist(self.task.whitelist)
                # images = target_node.get_images()
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
                # images = neighbor.get_images()
                images = neighbor.get_images_whitelist(self.task.whitelist)
                images = [os.path.join(root_path, image) for image in images]
                if images:
                    image_paths.extend(images)
                    image_labels.extend([len(visited) for _ in range(len(images))])
                    visited.add(neighbor.get_conceptnet_id())
                    log.info("Source class found: {}".format(neighbor.get_conceptnet_id()))

        Scads.close()

        transform = self.transform_image()
        train_data = CustomDataset(image_paths,
                                   labels=image_labels,
                                   transform=transform)

        return train_data, len(visited)

    def train(self, train_data, val_data):
        # Get Scads data and set up model
        scads_train_data, scads_num_classes = self._get_scads_data()
        log.info("Source classes found: {}".format(scads_num_classes))
        log.info("Number of source training images: {}".format(len(scads_train_data)))

        self.model = MultiTaskModel(self.model, len(self.task.classes),
                                    scads_num_classes, self.task.input_shape)

        super(MultiTaskTaglet, self).train(train_data, val_data)

    def _do_train(self, rank, q, train_data, val_data):
        # batch_size = min(len(train_data) // num_batches, 256)
        old_batch_size = self.batch_size
        self.batch_size = 128
        source_sampler = self._get_train_sampler(train_data, n_proc=self.n_proc, rank=rank)
        self.source_data_loader = self._get_dataloader(data=train_data, sampler=source_sampler)
        self.batch_size = old_batch_size

        super(MultiTaskTaglet, self)._do_train(rank, q, train_data, val_data)

    def _train_epoch(self, rank, train_data_loader):
        self.model.train()
        running_loss = 0
        running_acc = 0
        for source_batch, target_batch in zip(self.source_data_loader, train_data_loader):
            source_inputs, source_labels = source_batch
            target_inputs, target_labels = target_batch
            if self.use_gpu:
                source_inputs = source_inputs.cuda(rank)
                source_labels = source_labels.cuda(rank)
                target_inputs = target_inputs.cuda(rank)
                target_labels = target_labels.cuda(rank)

            self.optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                source_outputs = self.model.forward_source(source_inputs)
                source_loss = self.criterion(source_outputs, source_labels)
                target_outputs = self.model(target_inputs)
                target_loss = self.criterion(target_outputs, target_labels)

                loss = source_loss + target_loss
                loss.backward()
                self.optimizer.step()

            running_loss += loss.item()
            running_acc += self._get_train_acc(source_outputs, source_labels)
            running_acc += self._get_train_acc(target_outputs, target_labels)

        if not len(train_data_loader.dataset):
            return 0, 0

        epoch_loss = running_loss / len(train_data_loader.dataset)
        total_len = len(train_data_loader.dataset)
        total_len += len(self.source_data_loader.dataset)
        epoch_acc = running_acc.item() / total_len

        return epoch_loss, epoch_acc
