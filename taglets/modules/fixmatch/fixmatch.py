from taglets.modules.module import Module
from taglets.pipeline import Taglet
from .utils import TransformFixMatch
from copy import deepcopy
from enum import Enum

#from ..
#from ....data.custom_dataset import CustomDataset
#from ....pipeline import Cache, Taglet
#from ....scads import Scads, ScadsEmbedding

import math
import pickle


import torch
import logging
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F

import torchvision.transforms as transforms


log = logging.getLogger(__name__)


class Optimizer(Enum):
    SGD = 1
    ADAM = 2


# support for exponential moving average
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


class FixMatchModule(Module):
    def __init__(self, task):
        super().__init__(task)
        self.taglets = [FixMatchTaglet(task, use_ema=False, verbose=True)]


class FixMatchTaglet(Taglet):
    def __init__(self, task, steps_per_epoch=-1,
                             conf_thresh=0.95,
                             lambda_u=1,
                             nesterov=True,
                             mu=1,
                             weight_decay=0.01,
                             temp=0.95,
                             use_ema=False,
                             ema_decay=0.999,
                             optimizer=Optimizer.ADAM,
                             verbose=False):
        self.name = 'fixmatch'

        self.steps_per_epoch = steps_per_epoch
        self.conf_thresh = conf_thresh
        self.lambda_u = lambda_u
        self.nesterov = nesterov
        
        self.mu = mu
        self.weight_decay = weight_decay

        # temp used to sharpen logits
        self.temp = temp
        self.use_ema = use_ema

        self.org_unlabeled_transform = None

        if verbose:
            log.info('Initializing FixMatch with hyperparameters:')
            log.info('confidence threshold: %.4f', self.conf_thresh)
            log.info('nesterov: ' + str(self.nesterov))
            log.info("unlabeled loss weight (lambda u): %.4f", self.lambda_u)
            log.info('temperature: %.4f', self.temp)

        super().__init__(task)
        
        output_shape = self._get_model_output_shape(self.task.input_shape, self.model)
        self.model = torch.nn.Sequential(self.model,
                                         torch.nn.Linear(output_shape, len(self.task.classes)))
        self.lr = 0.001
        self.num_epochs = 200

        if use_ema:
            self.ema_model = ModelEMA(self.model, decay=ema_decay)

        self.unlabeled_batch_size = math.floor(self.mu * self.batch_size)
        if self.unlabeled_batch_size == 0:
            raise ValueError("unlabeled dataset is too small for FixMatch.")

        # according to paper, SGD results in better performance than ADAM
        self.opt_type = optimizer
        if self.opt_type == Optimizer.SGD:
            self.optimizer = torch.optim.SGD(self._params_to_update, lr=self.lr,
                                                                     momentum=0.9,
                                                                     nesterov=self.nesterov)

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
        else:
            self.model = self.model.cpu()
            self.model = nn.parallel.DistributedDataParallel(
                self.model, device_ids=None
            )

            if self.use_ema:
                self.ema_model.ema = self.ema_model.ema.cpu()

        # Creates distributed data loaders from datasets
        train_sampler = self._get_train_sampler(train_data, n_proc=self.n_proc, rank=rank)
        train_data_loader = self._get_dataloader(data=train_data, sampler=train_sampler,
                                                                  batch_size=self.batch_size)

        # batch size can't be larger than number of examples
        self.unlabeled_batch_size = min(self.unlabeled_batch_size, len(unlabeled_data))
        unlabeled_sampler = self._get_train_sampler(unlabeled_data, n_proc=self.n_proc, rank=rank)

        # copy unlabeled dataset to prevent adverse side effects
        unlabeled_data = deepcopy(unlabeled_data)

        fixmatch_transform = TransformFixMatch(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225],
                                                     input_shape=self.task.input_shape)

        # replace default transform with FixMatch Transform
        if not hasattr(unlabeled_data, "transform"):
            if not hasattr(unlabeled_data, "dataset"):
                raise ValueError("Invalid dataset. FixMatch cannot modify data transformer.")
            unlabeled_data.dataset.transform = fixmatch_transform
        else:
            unlabeled_data.transform = fixmatch_transform

        unlabeled_data_loader = self._get_dataloader(data=unlabeled_data,
                                                     sampler=unlabeled_sampler,
                                                     batch_size=self.unlabeled_batch_size)

        if self.steps_per_epoch == -1:
            self.steps_per_epoch = max(len(unlabeled_data_loader), len(unlabeled_data_loader))

        if self.opt_type == Optimizer.SGD:
            total_steps = self.steps_per_epoch * self.num_epochs
            self.lr_scheduler = get_cosine_schedule_with_warmup(self.optimizer,
                                                                num_warmup_steps=0,
                                                                num_training_steps=total_steps)

        if val_data is None:
            val_data_loader = None
        else:
            val_sampler = self._get_val_sampler(val_data, n_proc=self.n_proc, rank=rank)
            val_data_loader = self._get_dataloader(data=val_data, sampler=val_sampler,
                                                                  batch_size=self.batch_size)

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

            if self.opt_type == Optimizer.ADAM:
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
        unlabeled_data.transform = self.org_unlabeled_transform

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

    """
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
    """

    def _get_pred_classifier(self):
        return self.ema_model.ema if self.use_ema else self.model

    def _train_epoch(self, rank, train_data_loader, unlabeled_data_loader=None):
        self.model.train()

        #if self.use_ema:
        #    self.ema_model.ema.train()

        labeled_iter = iter(train_data_loader)
        unlabeled_iter = iter(unlabeled_data_loader)

        running_loss = 0.0
        running_acc = 0.0
        for i in range(len(train_data_loader)):
            try:
                inputs_x, targets_x = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(train_data_loader)
                inputs_x, targets_x = next(labeled_iter)

            try:
                # u_w = weak aug examples; u_s = strong aug examples
                ex = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_data_loader)
                ex = next(unlabeled_iter)

            try:
                inputs_u_w, inputs_u_s = ex[0], ex[1]
            except TypeError:
                raise ValueError("Unlabeled transform is not configured correctly.")

            batch_size = inputs_x.shape[0]
            inputs = torch.cat((inputs_x, inputs_u_w, inputs_u_s))
            inputs = inputs.cuda(rank) if self.use_gpu else inputs.cpu()
            targets_x = targets_x.cuda(rank) if self.use_gpu else targets_x.cpu()

            self.optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                logits = self.model(inputs)
                logits_x = logits[:batch_size]
                logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
                del logits

                lx = F.cross_entropy(logits_x, targets_x, reduction='mean')

                pseudo_label = torch.softmax(logits_u_w.detach() / self.temp, dim=-1)
                max_probs, targets_u = torch.max(pseudo_label, dim=-1)
                # binary mask to ignore unconfident psudolabels
                mask = max_probs.ge(self.conf_thresh).float()

                lu = (F.cross_entropy(logits_u_s, targets_u, reduction='none') * mask).mean()
                loss = lx + self.lambda_u * lu
                loss.backward()
                self.optimizer.step()

            if self.opt_type == Optimizer.SGD:
                self.lr_scheduler.step()
            if self.use_ema:
                self.ema_model.update(self.model)

            running_loss += loss.item()
            running_acc += self._get_train_acc(logits_x, targets_x)

        epoch_loss = running_loss / self.steps_per_epoch
        epoch_acc = running_acc.item() / len(unlabeled_data_loader.dataset)
        return epoch_loss, epoch_acc

