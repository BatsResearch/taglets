from taglets.modules.module import Module
from taglets.pipeline import Cache, Taglet

from .utils import TransformFixMatch

from copy import deepcopy

import os
import random
import torch
import logging
import numpy as np
import torchvision.transforms as transforms
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
import math
import pickle

log = logging.getLogger(__name__)


class ModelEMA(object):
    def __init__(self, model, decay):
        self.ema = deepcopy(model)
        self.ema.eval()
        self.decay = decay
        self.ema_has_module = hasattr(self.ema, 'module')
        self.param_keys = [k for k, _ in self.ema.named_parameters()]
        self.buffer_keys = [k for k, _ in self.ema.named_buffers()]
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        needs_module = hasattr(model, 'module') and not self.ema_has_module
        with torch.no_grad():
            msd = model.state_dict()
            esd = self.ema.state_dict()
            for k in self.param_keys:
                if needs_module:
                    j = 'module.' + k
                else:
                    j = k
                model_v = msd[j].detach()
                ema_v = esd[k]
                esd[k].copy_(ema_v * self.decay + (1. - self.decay) * model_v)

            for k in self.buffer_keys:
                if needs_module:
                    j = 'module.' + k
                else:
                    j = k
                esd[k].copy_(msd[j])


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda, last_epoch)


def interleave(x, size):
    s = list(x.shape)
    print(s)
    print([-1, size] + s[1:])
    print(x.reshape([-1, size] + s[1:]).shape)
    print(x.reshape([-1, size] + s[1:]).transpose(0, 1).shape)
    print(x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:]).shape)
    exit(1)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


class FixMatchModule(Module):
    def __init__(self, task):
        super().__init__(task)
        self.taglets = [FixMatchTaglet(task, verbose=True)]


class FixMatchTaglet(Taglet):
    def __init__(self, task, steps_per_epoch=-1,
                             conf_thresh=0.95,
                             lambda_u=1,
                             nesterov=True,
                             mu=1,
                             weight_decay=5e-4,
                             temp=0.95,
                             use_ema=False,
                             ema_decay=0.999,
                             verbose=False):
        self.name = 'fixmatch'

        self.steps_per_epoch = steps_per_epoch
        self.conf_thresh = conf_thresh
        self.lambda_u = lambda_u
        self.nesterov = nesterov

        self.lr = 1e-3
        self.mu = mu
        self.weight_decay = weight_decay

        # temp used to sharpen logits
        self.temp = temp
        self.use_ema = use_ema
        self.ema_decay = ema_decay

        if verbose:
            log.info('Initializing FixMatch with hyperparameters:')
            log.info('confidence threshold: %.4f', self.conf_thresh)
            log.info('nesterov: ' + str(self.nesterov))
            log.info("unlabeled loss weight (lambda u): %.4f", self.lambda_u)
            log.info('temperature: %.4f', self.temp)

        super().__init__(task)

        if use_ema:
            self.ema_model = ModelEMA(self.model, decay=self.ema_decay)

        self.unlabeled_batch_size = math.floor(self.mu * self.batch_size)
        if self.unlabeled_batch_size == 0:
            raise ValueError("unlabeled dataset is too small for FixMatch.")

        # according to paper, Adam results in poor performance compared to SGD
        # On basic tests, however, ordinary SGD appears to perform worse?
        #self.optimizer = torch.optim.SGD(self._params_to_update, lr=self.lr, momentum=0.9, nesterov=self.nesterov)

    def _do_train(self, rank, q, train_data, val_data, unlabeled_data=None):
        """
               One worker for training.

               This method carries out the actual training iterations. It is designed
               to be called by train().

               :param train_data: A dataset containing training data
               :param val_data: A dataset containing validation data
               :param unlabeled_data: A dataset containing unlabeled data
               :return:
               """

        if unlabeled_data is None:
            raise ValueError("Cannot train FixMatch taglet without unlabeled data.")

        if rank == 0:
            log.info('Beginning training')

        # Initializes distributed backend
        backend = 'nccl' if self.use_gpu else 'gloo'
        dist.init_process_group(
            backend=backend, init_method='env://', world_size=self.n_proc, rank=rank
        )

        # Configures model to be distributed
        if self.use_gpu:
            self.model = self.model.cuda(rank)
            self.model = nn.parallel.DistributedDataParallel(
                self.model, device_ids=[rank]
            )
            if self.use_ema:
                self.ema_model.ema = self.ema_model.ema.cuda(rank)
            #    self.ema_model.ema = nn.parallel.DistributedDataParallel(
            #        self.ema_model.ema, device_ids=[rank]
            #)
        else:
            self.model = self.model.cpu()
            self.model = nn.parallel.DistributedDataParallel(
                self.model, device_ids=None
            )

            if self.use_ema:
                self.ema_model.ema = self.ema_model.ema.cpu()
                #self.ema_model.ema = nn.parallel.DistributedDataParallel(
                #    self.ema_model.ema, device_ids=None
                #)

        # Creates distributed data loaders from datasets
        train_sampler = self._get_train_sampler(train_data, n_proc=self.n_proc, rank=rank)
        train_data_loader = self._get_dataloader(data=train_data, sampler=train_sampler, batch_size=self.batch_size)

        self.unlabeled_batch_size = min(self.unlabeled_batch_size, len(unlabeled_data))
        unlabeled_sampler = self._get_train_sampler(unlabeled_data, n_proc=self.n_proc, rank=rank)

        # replace default transform with FixMatch Transform
        unlabeled_data.transform = TransformFixMatch(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225],
                                                     input_shape=self.task.input_shape)
        unlabeled_data_loader = self._get_dataloader(data=unlabeled_data, sampler=unlabeled_sampler,
                                                                          batch_size=self.unlabeled_batch_size)

        if self.steps_per_epoch == -1:
            self.steps_per_epoch = max(len(unlabeled_data_loader), len(unlabeled_data_loader))
        #self.lr_scheduler = get_cosine_schedule_with_warmup(self.optimizer, 0, self.steps_per_epoch * self.num_epochs)

        if val_data is None:
            val_data_loader = None
        else:
            val_sampler = self._get_val_sampler(val_data, n_proc=self.n_proc, rank=rank)
            val_data_loader = self._get_dataloader(data=val_data, sampler=val_sampler, batch_size=self.batch_size)

        # Initializes statistics containers (will only be filled by lead process)
        best_model_to_save = None
        best_val_acc = 0
        train_loss_list = []
        train_acc_list = []
        val_loss_list = []
        val_acc_list = []

        # Iterates over epochs
        for epoch in range(self.num_epochs):
            if rank == 0:
                log.info("Epoch {}: ".format(epoch + 1))

            # this is necessary for shuffle to work
            train_sampler.set_epoch(epoch)

            # Trains on training data
            train_loss, train_acc = self._train_epoch(rank, train_data_loader, unlabeled_data_loader)

            # Evaluates on validation data
            if val_data_loader:
                val_loss, val_acc = self._validate_epoch(rank, val_data_loader)
            else:
                val_loss = 0
                val_acc = 0

            # Gathers results statistics to lead process
            summaries = [train_loss, train_acc, val_loss, val_acc]
            summaries = torch.tensor(summaries, requires_grad=False)
            if self.use_gpu:
                summaries = summaries.cuda(rank)
            else:
                summaries = summaries.cpu()
            dist.reduce(summaries, 0, op=dist.ReduceOp.SUM)
            train_loss, train_acc, val_loss, val_acc = summaries

            # Processes results if lead process
            if rank == 0:
                log.info('Train loss: {:.4f}'.format(train_loss))
                log.info('Train acc: {:.4f}%'.format(train_acc * 100))
                train_loss_list.append(train_loss)
                train_acc_list.append(train_acc)
                log.info('Validation loss: {:.4f}'.format(val_loss))
                log.info('Validation acc: {:.4f}%'.format(val_acc * 100))
                val_loss_list.append(val_loss)
                val_acc_list.append(val_acc)
                if val_acc > best_val_acc:
                    log.debug("Deep copying new best model." +
                              "(validation of {:.4f}%, over {:.4f}%)".format(
                                  val_acc * 100, best_val_acc * 100))
                    best_model_to_save = deepcopy(self.model.module.state_dict())
                    best_val_acc = val_acc
                    if self.save_dir:
                        torch.save(best_model_to_save, self.save_dir + '/model.pth.tar')

            if self.lr_scheduler:
                self.lr_scheduler.step()

        # Lead process saves plots and returns best model
        if rank == 0:
            if self.save_dir:
                val_dic = {'train': train_loss_list, 'validation': val_loss_list}
                self.save_plot('loss', val_dic, self.save_dir)
                val_dic = {'train': train_acc_list, 'validation': val_acc_list}
                self.save_plot('accuracy', val_dic, self.save_dir)
            if self.select_on_val and best_model_to_save is not None:
                self.model.module.load_state_dict(best_model_to_save)

            self.model.cpu()
            state_dict = self.model.module.state_dict()
            state_dict = pickle.dumps(state_dict)
            q.put(state_dict)

        # Use a barrier to keep all workers alive until they all finish,
        # due to shared CUDA tensors. See
        # https://pytorch.org/docs/stable/multiprocessing.html#multiprocessing-cuda-sharing-details
        dist.barrier()

    def _get_pred_classifier(self):
        return self.ema_model.ema if self.use_ema else self.model

    def _train_epoch(self, rank, train_data_loader, unlabeled_data_loader=None):
        self.model.train()

        labeled_iter = iter(train_data_loader)
        unlabeled_iter = iter(unlabeled_data_loader)

        running_loss = 0.0
        running_acc = 0.0
        for i in range(len(train_data_loader)):
            try:
                inputs_x, targets_x = next(labeled_iter)
            except:
                labeled_iter = iter(train_data_loader)
                inputs_x, targets_x = next(labeled_iter)

            try:
                (inputs_u_w, inputs_u_s) = next(unlabeled_iter)
            except:
                unlabeled_iter = iter(unlabeled_data_loader)
                (inputs_u_w, inputs_u_s) = next(unlabeled_iter)

            batch_size = inputs_x.shape[0]
            inputs = torch.cat((inputs_x, inputs_u_w, inputs_u_s))
            #inputs = interleave(
            #    torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2 * self.mu + 1)
            inputs = inputs.cuda(rank) if self.use_gpu else inputs.cpu()
            targets_x = targets_x.cuda(rank) if self.use_gpu else targets_x.cpu()

            self.optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                logits = self.model(inputs)
                #logits = de_interleave(logits, 2 * self.mu + 1)
                logits_x = logits[:batch_size]
                logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
                del logits

                Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')

                pseudo_label = torch.softmax(logits_u_w.detach() / self.temp, dim=-1)
                max_probs, targets_u = torch.max(pseudo_label, dim=-1)
                mask = max_probs.ge(self.conf_thresh).float()

                Lu = (F.cross_entropy(logits_u_s, targets_u, reduction='none') * mask).mean()

                loss = Lx + self.lambda_u * Lu
                loss.backward()
                self.optimizer.step()

            #self.lr_scheduler.step()
            if self.use_ema:
                self.ema_model.update(self.model)

            running_loss += loss.item()
            running_acc += self._get_train_acc(logits_x, targets_x)

        epoch_loss = running_loss / self.steps_per_epoch
        epoch_acc = running_acc.item() / len(train_data_loader.dataset)
        return epoch_loss, epoch_acc

    def _validate_epoch(self, rank, val_data_loader):
        """
        Validate for one epoch.
        :param val_data_loader: A dataloader containing validation data
        :param use_gpu: Whether or not to use the GPU
        :return: None
        """
        test_model = self.model
        if self.use_ema:
            test_model = self.ema_model.ema

        test_model.eval()
        running_loss = 0
        running_acc = 0
        for batch in val_data_loader:
            inputs = batch[0]
            labels = batch[1]
            if self.use_gpu:
                inputs = inputs.cuda(rank)
                labels = labels.cuda(rank)
            with torch.set_grad_enabled(False):
                outputs = test_model(inputs)
                loss = torch.nn.functional.cross_entropy(outputs, labels)
                _, preds = torch.max(outputs, 1)

            running_loss += loss.item()
            running_acc += torch.sum(preds == labels)

        epoch_loss = running_loss / len(val_data_loader.dataset)
        epoch_acc = running_acc.item() / len(val_data_loader.dataset)
        return epoch_loss, epoch_acc
