from .module import Module
from ..data.custom_dataset import CustomDataset
from ..pipeline import Cache, Taglet
from ..scads import Scads, ScadsEmbedding

import os
import random
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
        output_shape = self._get_model_output_shape(input_shape, self.base)
        self.fc_target = torch.nn.Linear(output_shape, self.num_target)
        self.fc_source = torch.nn.Linear(output_shape, self.num_source)

    def forward(self, target_inputs, source_inputs=None):
        x = self.base(target_inputs)
        x = torch.flatten(x, 1)
        target_outputs = self.fc_target(x)
        if source_inputs is None:
            return target_outputs
        else:
            x = self.base(source_inputs)
            x = torch.flatten(x, 1)
            source_outputs = self.fc_source(x)
            return target_outputs, source_outputs

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
        self.num_epochs = 50 if not os.environ.get("CI") else 5
        self.save_dir = os.path.join('trained_models', self.name)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.source_data = None

        self.img_per_related_class = 600 if not os.environ.get("CI") else 1
        self.num_related_class = 5

    def transform_image(self, train=True):
        """
        Get the transform to be used on an image.
        :return: A transform
        """
        data_mean = [0.485, 0.456, 0.406]
        data_std = [0.229, 0.224, 0.225]
        
        if train:
            return transforms.Compose([
                transforms.RandomResizedCrop(self.task.input_shape, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=data_mean, std=data_std)
            ])
        else:
            return transforms.Compose([
                transforms.Resize(self.task.input_shape),
                transforms.ToTensor(),
                transforms.Normalize(mean=data_mean, std=data_std)
            ])

    def _get_scads_data(self):
        data = Cache.get("scads", self.task.classes)
        if data is not None:
            image_paths, image_labels, all_related_class = data
        else:
            root_path = Scads.get_root_path()
            Scads.open(self.task.scads_path)
            ScadsEmbedding.load(self.task.scads_embedding_path)
            image_paths = []
            image_labels = []
            visited = set()

            def get_images(node, label):
                if node.get_conceptnet_id() not in visited:
                    visited.add(node.get_conceptnet_id())
                    images = node.get_images_whitelist(self.task.whitelist)
                    if len(images) < self.img_per_related_class:
                        return False
                    images = random.sample(images, self.img_per_related_class)
                    images = [os.path.join(root_path, image) for image in images]
                    image_paths.extend(images)
                    image_labels.extend([label] * len(images))
                    log.debug("Source class found: {}".format(node.get_conceptnet_id()))
                    return True
                return False

            all_related_class = 0
            for conceptnet_id in self.task.classes:
                cur_related_class = 0
                target_node = Scads.get_node_by_conceptnet_id(conceptnet_id)
                if get_images(target_node, all_related_class):
                    cur_related_class += 1
                    all_related_class += 1

                neighbors = ScadsEmbedding.get_related_nodes(target_node, self.num_related_class * 100)
                for neighbor in neighbors:
                    if get_images(neighbor, all_related_class):
                        cur_related_class += 1
                        all_related_class += 1
                        if cur_related_class >= self.num_related_class:
                            break

            Scads.close()
            Cache.set('scads', self.task.classes,
                      (image_paths, image_labels, all_related_class))

        transform = self.transform_image(train=True)
        train_data = CustomDataset(image_paths,
                                   labels=image_labels,
                                   transform=transform)

        return train_data, all_related_class

    def train(self, train_data, val_data):
        # Get Scads data and set up model
        self.source_data, num_classes = self._get_scads_data()
        log.info("Source classes found: {}".format(num_classes))
        log.info("Number of source training images: {}".format(len(self.source_data)))
        
        if num_classes == 0:
            self.valid = False
            return

        self.model = MultiTaskModel(self.model, len(self.task.classes),
                                    num_classes, self.task.input_shape)

        params_to_update = []
        for param in self.model.parameters():
            if param.requires_grad:
                params_to_update.append(param)
        self._params_to_update = params_to_update
        self.optimizer = torch.optim.Adam(self._params_to_update, lr=self.lr, weight_decay=1e-4)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.1)
        
        super(MultiTaskTaglet, self).train(train_data, val_data)

    def _do_train(self, rank, q, train_data, val_data):
        # batch_size = min(len(train_data) // num_batches, 256)
        old_batch_size = self.batch_size
        self.batch_size = 128
        source_sampler = self._get_train_sampler(self.source_data, n_proc=self.n_proc, rank=rank)
        self.source_data_loader = self._get_dataloader(data=self.source_data, sampler=source_sampler)
        self.batch_size = old_batch_size

        old_batch_size = self.batch_size
        self.batch_size = 8
        super(MultiTaskTaglet, self)._do_train(rank, q, train_data, val_data)
        self.batch_size = old_batch_size

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
                outputs = self.model(target_inputs, source_inputs)
                target_outputs, source_outputs = outputs
                source_loss = self.criterion(source_outputs, source_labels)
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
