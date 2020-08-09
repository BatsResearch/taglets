from taglets.data.custom_dataset import CustomDataset
from torch.utils import data

from .module import Module
from ..pipeline import Taglet
from ..scads import Scads, ScadsEmbedding

import os
import random
import torch
import logging
import copy
import numpy as np
import torchvision.transforms as transforms
import torch.nn as nn

log = logging.getLogger(__name__)

class MultiTaskModel(nn.Module):
    def __init__(self, model, num_target, num_source,input_shape):
        super().__init__()
        self.model = model
        # self.model.fc = torch.nn.Linear(self.model.fc.in_features, self.model.fc.in_features)
        self.num_target= num_target
        self.num_source = num_source
        self.base = nn.Sequential(*list(self.model.children())[:-1])
        # m = torch.nn.Sequential(*list(self.model.children())[:-1])
        output_shape = self._get_model_output_shape(input_shape, self.model)
        self.fc_target = torch.nn.Linear(output_shape, self.num_target)
        self.fc_source = torch.nn.Linear(output_shape, self.num_source)

    def forward(self, x):
        # x = self.model(x)
        # return self.fc_target(x), self.fc_source(x)

        x = self.base(x)
        x = torch.flatten(x, 1)
        # return self.fc_target(x), self.fc_source(x)
        clf_outputs = {}
        clf_outputs["fc_target"] = self.fc_target(x)
        clf_outputs["fc_source"] = self.fc_source(x)

        return clf_outputs

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

        self.img_per_related_class = 600
        self.num_related_class = 5

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
        ScadsEmbedding.load('predefined/numberbatch-en19.08.txt.gz')
        image_paths = []
        image_labels = []
        visited = set()

        def get_images(node):
            if node.get_conceptnet_id() not in visited:
                visited.add(node.get_conceptnet_id())
                images = node.get_images_whitelist(self.task.whitelist)
                if len(images) < self.img_per_related_class:
                    return False
                images = random.sample(images, self.img_per_related_class)
                images = [os.path.join(root_path, image) for image in images]
                image_paths.extend(images)
                image_labels.extend([len(visited) for _ in range(len(images))])
                log.debug("Source class found: {}".format(node.get_conceptnet_id()))
                return True
            return False

        for conceptnet_id in self.task.classes:
            cur_related_class = 0
            target_node = Scads.get_node_by_conceptnet_id(conceptnet_id)
            if get_images(target_node):
                cur_related_class += 1
    
            neighbors = ScadsEmbedding.get_related_nodes(target_node, self.num_related_class * 100)
            for neighbor in neighbors:
                if get_images(neighbor):
                    cur_related_class += 1
                    if cur_related_class >= self.num_related_class:
                        break

        Scads.close()

        transform = self.transform_image()
        train_data = CustomDataset(image_paths,
                                   labels=image_labels,
                                   transform=transform)

        # batch_size = min(len(train_data) // num_batches, 256)
        batch_size = 128
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
        log.info("number of source training images: {}".format(len(scads_train_data_loader.dataset)))
        self.model = MultiTaskModel(self.model, len(self.task.classes), scads_num_classes,self.task.input_shape)


        # Parameters needed to be updated based on freezing layer
        params_to_update = []
        for param in self.model.parameters():
            if param.requires_grad:
                params_to_update.append(param)
        self._params_to_update = params_to_update
        self.optimizer = torch.optim.Adam(self._params_to_update, lr=self.lr, weight_decay=1e-4)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

        # Train
        log.info('Beginning training')

        if use_gpu:
            self.model = self.model.cuda()
        else:
            self.model = self.model.cpu()

        best_model_to_save = None
        source_train_loss_list = []
        target_train_loss_list = []
        source_train_acc_list = []
        target_train_acc_list = []
        val_loss_list = []
        val_acc_list = []

        for epoch in range(self.num_epochs):
            log.info('Epoch: %s', epoch)

            # Train on training data
            epoch_loss_source_train, epoch_loss_target_train, epoch_acc_source_train, epoch_acc_target_train = self._multitask_train_epoch(train_data_loader,
                                                                scads_train_data_loader,
                                                                use_gpu)
            source_train_loss_list.append(epoch_loss_source_train)
            target_train_loss_list.append(epoch_loss_target_train)

            source_train_acc_list.append(epoch_acc_source_train)
            target_train_acc_list.append(epoch_acc_target_train)

            log.info("Epoch {} result: ".format(epoch + 1))
            log.info('source train loss: {:.4f}'.format(epoch_loss_source_train))
            log.info('source train acc: {:.4f}%'.format(epoch_acc_source_train*100))
            log.info('target train loss: {:.4f}'.format(epoch_loss_target_train))
            log.info('target train acc: {:.4f}%'.format(epoch_acc_target_train*100))

            # Evaluation on validation data
            if not val_data_loader:
                val_loss = 0
                val_acc = 0
                continue
            val_loss, val_acc = self._validate_epoch(val_data_loader, use_gpu)
            val_loss_list.append(val_loss)
            val_acc_list.append(val_acc)
            log.info('target validation loss: {:.4f}'.format(val_loss))
            log.info('target validation acc: {:.4f}%'.format(val_acc*100))

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

        val_dic = {'target_train': target_train_loss_list, 'validation': val_loss_list}
        if self.save_dir:
            self.save_plot('loss', val_dic, self.save_dir)
        val_dic = {'target_train': target_train_acc_list, 'validation': val_acc_list}
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
        running_loss_source_train = 0.0
        running_loss_target_train = 0.0

        running_corrects_source_train = 0
        running_corrects_target_train = 0
        # for batch_idx, (target_batch, source_batch) in enumerate(zip(target_data_loader, scads_data_loader)):
        for s, t in zip(scads_data_loader, target_data_loader):

            input_source, label_source = s
            input_target, label_target = t
            if use_gpu:
                input_source = input_source.cuda()
                label_source = label_source.cuda()
                input_target = input_target.cuda()
                label_target = label_target.cuda()


            self.optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                # ### update using source data
                self.model.fc_target.train(False)
                self.model.fc_source.train(True)
                self.model.fc_target.weight.requires_grad = False
                self.model.fc_target.bias.requires_grad = False
                self.model.fc_source.weight.requires_grad = True
                self.model.fc_source.bias.requires_grad = True

                assert self.model.fc_target.weight.requires_grad == False
                assert self.model.fc_source.weight.requires_grad == True
                output_source = self.model(input_source)
                loss_source = self.criterion(output_source['fc_source'], label_source)
                _, pred_source = torch.max(output_source['fc_source'], 1)
                loss_source.backward()
                self.optimizer.step()

                ####### update using target data
                self.model.fc_target.train(True)
                self.model.fc_source.train(False)

                self.model.fc_target.weight.requires_grad = True
                self.model.fc_target.bias.requires_grad = True
                self.model.fc_source.weight.requires_grad = False
                self.model.fc_source.bias.requires_grad = False

                assert self.model.fc_target.weight.requires_grad == True
                assert self.model.fc_source.weight.requires_grad == False

                output_target = self.model(input_target)
                loss_target = self.criterion(output_target['fc_target'], label_target)

                _, pred_target = torch.max(output_target['fc_target'], 1)
                loss_target.backward()
                self.optimizer.step()

            running_loss_source_train += loss_source.item() * input_source.size(0)
            running_corrects_source_train += torch.sum(pred_source == label_source.data)

            running_loss_target_train += loss_target.item() * input_target.size(0)
            running_corrects_target_train += torch.sum(pred_target == label_target.data)

        if not len(target_data_loader.dataset):
            return 0, 0

        # epoch_loss = running_loss / len(target_data_loader.dataset)
        # epoch_acc = running_acc.item() / len(target_data_loader.dataset)
        epoch_loss_source_train = running_loss_source_train / len(scads_data_loader.dataset)
        epoch_acc_source_train = running_corrects_source_train.double() / len(scads_data_loader.dataset)

        epoch_loss_target_train = running_loss_target_train / len(target_data_loader.dataset)
        epoch_acc_target_train = running_corrects_target_train.double() / len(target_data_loader.dataset)

        return epoch_loss_source_train, epoch_loss_target_train,epoch_acc_source_train,epoch_acc_target_train

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
                outputs = self.model(inputs)
                loss = self.criterion(outputs['fc_target'], labels)
                _, preds = torch.max(outputs['fc_target'], 1)

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
                outputs = self.model(inputs)
                _, preds = torch.max(outputs['fc_target'], 1)
                predicted_labels = predicted_labels + preds.detach().cpu().tolist()
        return predicted_labels