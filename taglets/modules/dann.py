from .module import Module
from ..data.custom_dataset import CustomImageDataset
from ..pipeline import Cache, ImageTagletWithAuxData
from ..scads import Scads, ScadsEmbedding

from accelerate import Accelerator
accelerator = Accelerator()
import os
import random
import torch
import logging
import numpy as np
import torchvision.transforms as transforms
import torch.nn as nn
import torch.autograd as autograd
from nltk.corpus import wordnet as wn

log = logging.getLogger(__name__)


class DannModel(nn.Module):
    def __init__(self, model, num_target, num_source, input_shape):
        super().__init__()
        self.base = nn.Sequential(*list(model.children())[:-1])
        output_shape = self._get_model_output_shape(input_shape, self.base)
        self.fc_target = torch.nn.Linear(output_shape, num_target)
        self.fc_source = torch.nn.Linear(output_shape, num_source)
        self.fc_domain = torch.nn.Linear(output_shape, 2)

    def forward(self, input, fc='target'):
        x = self.base(input)
        x = torch.flatten(x, 1)
        if fc == 'target':
            output = self.fc_target(x)
        elif fc == 'source':
            output = self.fc_source(x)
        else:
            output = self.fc_domain(x)
        return output

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


class DannModule(Module):
    """
    A module that pre-trains on datasets selected from the SCADS and then
    transfers to available labeled data
    """

    def __init__(self, task):
        super().__init__(task)
        self.taglets = [DannTaglet(task)]


class DannTaglet(ImageTagletWithAuxData):
    def __init__(self, task):
        super().__init__(task)
        self.name = 'dann'
        self.num_epochs = 8 if not os.environ.get("CI") else 5
        if os.getenv("LWLL_TA1_PROB_TASK") is not None:
            self.save_dir = os.path.join('/home/tagletuser/trained_models', self.name)
        else:
            self.save_dir = os.path.join('trained_models', self.name)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.source_data = None

        self.num_related_class = 5 if len(self.task.classes) < 100 else (3 if len(self.task.classes) < 300 else 1)
        self.training_first_stage = True
        self.epoch = 0
        
        self.batch_size = self.batch_size // 4
        self.unlabeled_batch_size = self.unlabeled_batch_size // 4

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
        data = Cache.get("scads-dann", self.task.classes)
        if data is not None:
            image_paths, image_labels, all_related_class = data
        else:
            root_path = Scads.get_root_path()
            Scads.open(self.task.scads_path)
            ScadsEmbedding.load(self.task.scads_embedding_path, self.task.processed_scads_embedding_path)
            image_paths = []
            image_labels = []
            visited = set()

            target_synsets = []
            for conceptnet_id in self.task.classes:
                class_name = conceptnet_id[6:]
                target_synsets = target_synsets + wn.synsets(class_name, pos='n')

            def get_images(node, label, is_neighbor):
                if is_neighbor and node.get_conceptnet_id() in self.task.classes:
                    return False
                if node.get_conceptnet_id() not in visited:
                    visited.add(node.get_conceptnet_id())
                
                    synsets = wn.synsets(node.get_conceptnet_id()[6:], pos='n')
                    for synset in synsets:
                        for target_synset in target_synsets:
                            lch = synset.lowest_common_hypernyms(target_synset)[0]
                            if target_synset.shortest_path_distance(lch) <= self.prune:
                                return False
                
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
            
            # Unlike other methods, for Dann, the auxiliary classes for each target class are merged

            all_related_class = 0
            for conceptnet_id in self.task.classes:
                cur_related_class = 0
                target_node = Scads.get_node_by_conceptnet_id(conceptnet_id)
                if get_images(target_node, all_related_class, False):
                    cur_related_class += 1

                ct = 1
                while cur_related_class < self.num_related_class:
                    processed_embeddings_exist = (self.task.processed_scads_embedding_path is not None)
                    neighbors = ScadsEmbedding.get_related_nodes(target_node,
                                                                 limit=self.num_related_class * 10 * ct,
                                                                 only_with_images=processed_embeddings_exist)
                    for neighbor in neighbors:
                        if get_images(neighbor, all_related_class, True):
                            cur_related_class += 1
                            if cur_related_class >= self.num_related_class:
                                break
                    ct = ct * 2
                all_related_class += 1
            Scads.close()
            Cache.set('scads-dann', self.task.classes,
                      (image_paths, image_labels, all_related_class))

        transform = self.transform_image(train=True)
        train_data = CustomImageDataset(image_paths,
                                        labels=image_labels,
                                        transform=transform)

        return train_data, all_related_class

    def train(self, train_data, val_data, unlabeled_data=None):
        # Get Scads data and set up model
        self.source_data, num_classes = self._get_scads_data()
        log.info("Source classes found: {}".format(num_classes))
        log.info("Number of source training images: {}".format(len(self.source_data)))
    
        if num_classes == 0:
            self.valid = False
            return

        self.model = DannModel(self.model, len(self.task.classes), num_classes, self.task.input_shape)

        params_to_update = []
        for param in self.model.parameters():
            if param.requires_grad:
                params_to_update.append(param)
        self._params_to_update = params_to_update
        self.optimizer = torch.optim.SGD(self._params_to_update, lr=0.003, momentum=0.9)
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[4, 6], gamma=0.1)

        if len(train_data) < 1024:
            num_duplicates = (1024 // len(train_data)) + 1
            train_data = torch.utils.data.ConcatDataset([train_data] * num_duplicates)
        
        super(DannTaglet, self).train(train_data, val_data, unlabeled_data)

    def _do_train(self, train_data, val_data, unlabeled_data=None):
        self.source_data_loader = self._get_dataloader(data=self.source_data, shuffle=True)
        super(DannTaglet, self)._do_train(train_data, val_data, unlabeled_data)

    def _train_epoch(self, train_data_loader, unlabeled_data_loader=None):
        self.model.train()
        running_loss = 0
        running_acc = 0
        total_len = 0
        iteration = 0
        data_iter = iter(train_data_loader)
        if unlabeled_data_loader:
            unlabeled_data_iter = iter(unlabeled_data_loader)
        for source_batch in self.source_data_loader:
            try:
                target_batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_data_loader)
                target_batch = next(data_iter)
            p = (iteration + self.epoch * len(self.source_data_loader)) / (self.num_epochs * len(self.source_data_loader))
            alpha = 2 / (1 + np.exp(-10 * p)) - 1
            iteration += 1
            source_inputs, source_labels = source_batch
            target_inputs, target_labels = target_batch
            source_zeros = torch.zeros(len(source_inputs), dtype=torch.long, device=source_inputs.device)
            target_zeros = torch.zeros(len(target_inputs), dtype=torch.long, device=target_inputs.device)
            target_ones = torch.ones(len(target_inputs), dtype=torch.long, device=target_inputs.device)
            if unlabeled_data_loader:
                try:
                    unlabeled_inputs = next(unlabeled_data_iter)
                except StopIteration:
                    unlabeled_data_iter = iter(unlabeled_data_loader)
                    unlabeled_inputs = next(unlabeled_data_iter)
                unlabeled_zeros = torch.zeros(len(unlabeled_inputs), dtype=torch.long, device=unlabeled_inputs.device)
                unlabeled_ones = torch.ones(len(unlabeled_inputs), dtype=torch.long, device=unlabeled_inputs.device)

            for param in self.model.backbone.parameters():
                param.requires_grad = True
            for param in self.model.fc_domain.parameters():
                param.requires_grad = False

            self.optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                source_classes = self.model(source_inputs, fc='source')
                target_classes = self.model(target_inputs, fc='target')
                target_domains = self.model(target_inputs, fc='domain')
                source_class_loss = self.criterion(source_classes, source_labels)
                target_class_loss = self.criterion(target_classes, target_labels)
                adv_target_domain_loss = self.criterion(target_domains, target_zeros)
                if unlabeled_data_loader:
                    unlabeled_target_domains = self.model(unlabeled_inputs, fc='domain')
                    adv_target_domain_loss += self.criterion(unlabeled_target_domains, unlabeled_zeros)
                loss = source_class_loss + target_class_loss + adv_target_domain_loss

                accelerator.backward(loss)
                self.optimizer.step()
                
            for param in self.model.backbone.parameters():
                param.requires_grad = False
            for param in self.model.fc_domain.parameters():
                param.requires_grad = True

            self.optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                source_domains = self.model(source_inputs, fc='domain')
                target_domains = self.model(target_inputs, fc='domain')
                source_domain_loss = self.criterion(source_domains, source_zeros)
                target_domain_loss = self.criterion(target_domains, target_ones)
                if unlabeled_data_loader:
                    unlabeled_target_domains = self.model(unlabeled_inputs, fc='domain')
                    target_domain_loss += self.criterion(unlabeled_target_domains, unlabeled_ones)
                loss = source_domain_loss + target_domain_loss
    
                accelerator.backward(loss)
                self.optimizer.step()

            target_classes = accelerator.gather(target_classes.detach())
            target_labels = accelerator.gather(target_labels)

            running_loss += loss.item()
            running_acc += self._get_train_acc(target_classes, target_labels).item()
            total_len += len(target_labels)
        self.epoch += 1
        if not len(train_data_loader.dataset):
            return 0, 0

        epoch_loss = running_loss / len(train_data_loader)
        epoch_acc = running_acc / total_len

        return epoch_loss, epoch_acc
