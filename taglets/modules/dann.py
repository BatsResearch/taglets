from .module import Module
from ..data.custom_dataset import CustomImageDataset
from ..pipeline import Cache, ImageTaglet
from ..scads import Scads, ScadsEmbedding

import os
import random
import torch
import logging
import numpy as np
import torchvision.transforms as transforms
import torch.nn as nn
import torch.autograd as autograd

log = logging.getLogger(__name__)


class GradientReversalLayer(autograd.Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class DannModel(nn.Module):
    def __init__(self, model, num_target, num_source, input_shape):
        super().__init__()
        self.base = nn.Sequential(*list(model.children())[:-1])
        output_shape = self._get_model_output_shape(input_shape, self.base)
        self.hidden_source = torch.nn.Linear(output_shape, output_shape)
        self.relu = torch.nn.ReLU()
        self.fc_source = torch.nn.Linear(output_shape, num_source)
        self.hidden_target = torch.nn.Linear(output_shape, output_shape)
        self.fc_target = torch.nn.Linear(output_shape, num_target)
        self.hidden_domain = torch.nn.Linear(output_shape, output_shape)
        self.fc_domain = torch.nn.Linear(output_shape, 2)

    def forward(self, target_input, source_input=None, unlabeled_input=None, alpha=1.0):
        x = self.base(target_input)
        x = torch.flatten(x, 1)
        target_class = self.fc_target(self.relu(self.hidden_target(x)))
        if source_input is None:
            return target_class
        reverse_x = GradientReversalLayer.apply(x, alpha)
        target_domain = self.fc_domain(self.relu(self.hidden_domain(reverse_x)))
        target_dist = (target_class, target_domain)

        x = self.base(source_input)
        x = torch.flatten(x, 1)
        source_class = self.fc_source(self.relu(self.hidden_source(x)))
        reverse_x = GradientReversalLayer.apply(x, alpha)
        source_domain = self.fc_domain(self.relu(self.hidden_domain(reverse_x)))
        source_dist = (source_class, source_domain)
        if unlabeled_input is None or not len(unlabeled_input):
            return target_dist, source_dist, None

        x = self.base(unlabeled_input)
        x = torch.flatten(x, 1)
        reverse_x = GradientReversalLayer.apply(x, alpha)
        unlabeled_domain = self.fc_domain(self.relu(self.hidden_domain(reverse_x)))
        return target_dist, source_dist, unlabeled_domain

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

    def _remove_extra_heads(self):
        self.hidden_source = None
        self.fc_source = None
        self.hidden_domain = None
        self.fc_domain = None


class DannModule(Module):
    """
    A module that pre-trains on datasets selected from the SCADS and then
    transfers to available labeled data
    """

    def __init__(self, task):
        super().__init__(task)
        self.taglets = [DannTaglet(task)]


class DannTaglet(ImageTaglet):
    def __init__(self, task):
        super().__init__(task)
        self.name = 'dann'
        self.num_epochs = 2 if not os.environ.get("CI") else 5
        if os.getenv("LWLL_TA1_PROB_TASK") is not None:
            self.save_dir = os.path.join('/home/tagletuser/trained_models', self.name)
        else:
            self.save_dir = os.path.join('trained_models', self.name)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.source_data = None

        self.img_per_related_class = 600 if not os.environ.get("CI") else 1
        self.num_related_class = 5
        self.training_first_stage = True
        self.epoch = 0

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
        train_data = CustomImageDataset(image_paths,
                                        labels=image_labels,
                                        transform=transform)

        return train_data, all_related_class

    def train(self, train_data, val_data, unlabeled_data=None):
        # Get Scads data and set up model
        self.source_data, num_classes = self._get_scads_data()
        self.source_data = train_data
        log.info("Source classes found: {}".format(num_classes))
        log.info("Number of source training images: {}".format(len(self.source_data)))

        if num_classes == 0:
            self.valid = False
            return

        self.model = DannModel(self.model, len(self.task.classes), num_classes, self.task.input_shape)

        # Domain adversarial training
        self._update_params()
        super(DannTaglet, self).train(train_data, val_data, unlabeled_data)

        # Finetune target data
        self.training_first_stage = False
        self.model._remove_extra_heads()
        self._update_params()
        super(DannTaglet, self).train(train_data, val_data, unlabeled_data)

    def _update_params(self):
        self.params_to_update = []
        for param in self.model.parameters():
            if param.requires_grad:
                self.params_to_update.append(param)
        self.optimizer = torch.optim.Adam(self._params_to_update, lr=self.lr, weight_decay=1e-4)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.1)

    def _do_train(self, rank, q, train_data, val_data, unlabeled_data=None):
        # batch_size = min(len(train_data) // num_batches, 256)
        if self.training_first_stage:
            old_batch_size = self.batch_size
            # Memory bottleneck if batch size is too large
            self.batch_size = max(int(old_batch_size/8), 8) if not os.environ.get("CI") else 32
            source_sampler = self._get_train_sampler(self.source_data, n_proc=self.n_proc, rank=rank)
            self.source_data_loader = self._get_dataloader(data=self.source_data, sampler=source_sampler)
            self.batch_size = old_batch_size
        old_batch_size = self.batch_size
        self.batch_size = max(int(old_batch_size/8), 8)
        super(DannTaglet, self)._do_train(rank, q, train_data, val_data, unlabeled_data)
        self.batch_size = old_batch_size

    def _train_epoch(self, rank, train_data_loader, unlabeled_data=None):
        if self.training_first_stage:
            return self._adapt(rank, train_data_loader, unlabeled_data)
        return super(DannTaglet, self)._train_epoch(rank, train_data_loader, unlabeled_data)

    def _adapt(self, rank, train_data_loader, unlabeled_data):
        self.model.train()
        running_loss = 0
        running_acc = 0
        total_len = 0
        if unlabeled_data:
            data = zip(self.source_data_loader, train_data_loader, unlabeled_data)
            dataloader_len = min(len(self.source_data_loader), len(train_data_loader), len(unlabeled_data))
        else:
            data = zip(self.source_data_loader, train_data_loader)
            dataloader_len = min(len(self.source_data_loader), len(train_data_loader))
        iteration = 0
        for source_batch, target_batch, unlabeled_inputs in data:
            p = (iteration + self.epoch * dataloader_len) / (self.num_epochs * dataloader_len)
            alpha = 2 / (1 + np.exp(-10 * p)) - 1
            iteration += 1
            source_inputs, source_labels = source_batch
            target_inputs, target_labels = target_batch
            zeros = torch.zeros(len(source_inputs), dtype=torch.long)
            ones = torch.ones(len(target_inputs), dtype=torch.long)
            if unlabeled_data:
                unlabeled_ones = torch.zeros(len(unlabeled_inputs), dtype=torch.long)
            if self.use_gpu:
                source_inputs = source_inputs.cuda(rank)
                source_labels = source_labels.cuda(rank)
                target_inputs = target_inputs.cuda(rank)
                target_labels = target_labels.cuda(rank)
                zeros = zeros.cuda(rank)
                ones = ones.cuda(rank)
                if unlabeled_data:
                    unlabeled_inputs = unlabeled_inputs.cuda(rank)
                    unlabeled_ones = unlabeled_ones.cuda(rank)

            self.optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                (target_classes, target_domains), (source_classes, source_domains), unlabeled_target_domains = self.model(
                    target_inputs,
                    source_inputs,
                    unlabeled_inputs,
                    alpha
                )
                source_class_loss = self.criterion(source_classes, source_labels)
                target_class_loss = self.criterion(target_classes, target_labels)
                source_domain_loss = self.criterion(source_domains, zeros)
                target_domain_loss = self.criterion(target_domains, ones)
                if unlabeled_target_domains is not None:
                    target_domain_loss += self.criterion(unlabeled_target_domains, unlabeled_ones)
                loss = source_class_loss + target_class_loss + source_domain_loss + target_domain_loss

                loss.backward()
                self.optimizer.step()

            running_loss += loss.item()
            running_acc += self._get_train_acc(source_classes, source_labels)
            total_len += len(source_labels)
        self.epoch += 1
        if not len(train_data_loader.dataset):
            return 0, 0

        epoch_loss = running_loss / len(train_data_loader.dataset)
        epoch_acc = running_acc.item() / total_len

        return epoch_loss, epoch_acc
