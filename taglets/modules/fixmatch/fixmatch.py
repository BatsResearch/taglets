import math
import os
import random
import torch
import logging
import torch.nn.functional as F
import torchvision.transforms as transforms
from copy import deepcopy
from enum import Enum
from accelerate import Accelerator
accelerator = Accelerator()

from ..module import Module
from ...data.custom_dataset import CustomImageDataset
from ...pipeline import Cache, ImageTaglet
from ...scads import Scads, ScadsEmbedding
from .utils import TransformFixMatch, is_grayscale

log = logging.getLogger(__name__)


class Optimizer(Enum):
    SGD = 1
    ADAM = 2


class ModelEMA(object):
    """
    ModelEMA is a layer over a Pytorch Module that implements exponential moving average (EMA).
    Note: EMA may result in worse performance, depending on the dataset you're training on.
    """

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


# Custom learning rate scheduler used in FixMatch
def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7. / 16.,
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
        self.taglets = [FixMatchTaglet(task, optimizer=Optimizer.ADAM, use_ema=False, verbose=False)]


class FixMatchTaglet(ImageTaglet):
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
                 verbose=False,
                 use_scads=True):
        self.name = 'fixmatch'

        self.steps_per_epoch = steps_per_epoch
        self.conf_thresh = conf_thresh
        self.lambda_u = lambda_u
        self.nesterov = nesterov
        self.mu = mu
        self.use_scads = use_scads

        self.img_per_related_class = 600 if not os.environ.get("CI") else 1
        self.num_related_class = 5

        # temp used to sharpen logits
        self.temp = temp
        self.use_ema = use_ema
        self.ema_decay = ema_decay

        self.org_unlabeled_transform = None

        if verbose:
            log.info('Initializing FixMatch with hyperparameters:')
            log.info('ema enabled: %s', self.use_ema)
            log.info('confidence threshold: %.4f', self.conf_thresh)
            log.info('nesterov: ' + str(self.nesterov))
            log.info("unlabeled loss weight (lambda u): %.4f", self.lambda_u)
            log.info('temperature: %.4f', self.temp)

        super().__init__(task)

        self.weight_decay = weight_decay
        
        # ratio of labeled data and unlabeled data is one-to-one
        self.batch_size = self.batch_size // 2
        self.unlabeled_batch_size = math.floor(self.mu * self.batch_size)
        if self.unlabeled_batch_size == 0:
            raise ValueError("unlabeled dataset is too small for FixMatch.")

        self.opt_type = optimizer
        
        if os.getenv("LWLL_TA1_PROB_TASK") is not None:
            self.save_dir = os.path.join('/home/tagletuser/trained_models', self.name)
        else:
            self.save_dir = os.path.join('trained_models', self.name)
        if not os.path.exists(self.save_dir) and accelerator.is_local_main_process:
            os.makedirs(self.save_dir)
        accelerator.wait_for_everyone()

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
        train_dataset = CustomImageDataset(image_paths,
                                            labels=image_labels,
                                            transform=transform)

        return train_dataset, all_related_class

    def _init_unlabeled_transform(self, unlabeled_data):
        if not hasattr(unlabeled_data, "transform"):
            if not hasattr(unlabeled_data, "dataset"):
                raise ValueError("Invalid dataset. FixMatch cannot modify data transformer.")
            unlabeled_data.dataset.transform = TransformFixMatch(mean=[0.485, 0.456, 0.406],
                                                                 std=[0.229, 0.224, 0.225],
                                                                 input_shape=self.task.input_shape,
                                                                 grayscale=is_grayscale(
                                                                     unlabeled_data.dataset.transform))
        else:
            unlabeled_data.transform = TransformFixMatch(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225],
                                                         input_shape=self.task.input_shape,
                                                         grayscale=is_grayscale(unlabeled_data.transform))

    def train(self, train_data, val_data, unlabeled_data=None):
        if self.task.scads_path is None:
            self.use_scads = False

        # warm-up using scads data
        if self.use_scads:
            scads_train_data, scads_num_classes = self._get_scads_data()

            encoder = torch.nn.Sequential(*list(self.model.children())[:-1])
            output_shape = self._get_model_output_shape(self.task.input_shape, encoder)
            self.model.fc = torch.nn.Linear(output_shape, scads_num_classes)

            params_to_update = []
            for param in self.model.parameters():
                if param.requires_grad:
                    params_to_update.append(param)
            self._params_to_update = params_to_update
            self.optimizer = torch.optim.Adam(self._params_to_update, lr=self.lr, weight_decay=1e-4)
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.1)

            batch_size_copy = self.batch_size
            num_epochs_copy = self.num_epochs
            use_ema_copy = self.use_ema

            self.batch_size = 2 * self.batch_size
            self.num_epochs = 25 if not os.environ.get("CI") else 5
            self.use_ema = False

            super(FixMatchTaglet, self).train(scads_train_data, None, None)

            self.batch_size = batch_size_copy
            self.num_epochs = num_epochs_copy
            self.use_ema = use_ema_copy
            self.use_scads = False

        # init fixmatch head
        encoder = torch.nn.Sequential(*list(self.model.children())[:-1])
        output_shape = self._get_model_output_shape(self.task.input_shape, encoder)
        self.model.fc = torch.nn.Linear(output_shape, len(self.task.classes))

        params_to_update = []
        for param in self.model.parameters():
            if param.requires_grad:
                params_to_update.append(param)
        self._params_to_update = params_to_update

        if self.opt_type == Optimizer.SGD:
            self.optimizer = torch.optim.SGD(self._params_to_update, lr=self.lr, momentum=0.9, nesterov=self.nesterov)
        else:
            self.optimizer = torch.optim.Adam(self._params_to_update, lr=self.lr, weight_decay=self.weight_decay)

        if self.use_ema:
            self.ema_model = ModelEMA(self.model, decay=self.ema_decay)

        # copy unlabeled dataset to prevent adverse side effects
        unlabeled_data = deepcopy(unlabeled_data)

        # replace default transform with FixMatch Transform
        self._init_unlabeled_transform(unlabeled_data)
        super(FixMatchTaglet, self).train(train_data, val_data, unlabeled_data)

    def _do_train(self, train_data, val_data, unlabeled_data=None):
        """
               One worker for training.

               This method carries out the actual training iterations. It is designed
               to be called by train().

               :param train_data: A dataset containing training data
               :param val_data: A dataset containing validation data
               :param unlabeled_data: A dataset containing unlabeled data
               :return:
               """
        log.info('Beginning training')

        train_data_loader = self._get_dataloader(data=train_data, shuffle=True,
                                                 batch_size=self.batch_size)

        if not self.use_scads:
            # batch size can't be larger than number of examples
            self.unlabeled_batch_size = min(self.unlabeled_batch_size, len(unlabeled_data))

            unlabeled_data_loader = self._get_dataloader(data=unlabeled_data,
                                                         shuffle=True,
                                                         batch_size=self.unlabeled_batch_size)

            if self.steps_per_epoch == -1:
                self.steps_per_epoch = max(len(train_data_loader), len(unlabeled_data_loader))

            if self.opt_type == Optimizer.SGD:
                total_steps = self.steps_per_epoch * self.num_epochs
                self.lr_scheduler = get_cosine_schedule_with_warmup(self.optimizer,
                                                                    num_warmup_steps=0,
                                                                    num_training_steps=total_steps)
        else:
            unlabeled_data_loader = None
            self.steps_per_epoch = len(train_data_loader)

        if val_data is None:
            val_data_loader = None
        else:
            val_data_loader = self._get_dataloader(data=val_data, shuffle=False,
                                                   batch_size=self.batch_size)

        # Initializes statistics containers (will only be filled by lead process)
        best_ema_model_to_save = None
        best_model_to_save = None
        best_val_acc = 0
        train_loss_list = []
        train_acc_list = []
        val_loss_list = []
        val_acc_list = []

        self.model, self.optimizer = accelerator.prepare(self.model, self.optimizer)

        # Iterates over epochs
        for epoch in range(self.num_epochs):
            log.info("Epoch {}: ".format(epoch + 1))

            # Trains on training data
            train_loss, train_acc = self._train_epoch(train_data_loader, unlabeled_data_loader)

            # Evaluates on validation data
            if val_data_loader:
                val_loss, val_acc = self._validate_epoch(val_data_loader)
            else:
                val_loss = 0
                val_acc = 0

            log.info('Train loss: {:.4f}'.format(train_loss))
            log.info('Train acc: {:.4f}%'.format(train_acc * 100))
            train_loss_list.append(train_loss)
            train_acc_list.append(train_acc)
            log.info('Validation loss: {:.4f}'.format(val_loss))
            log.info('Validation acc: {:.4f}%'.format(val_acc * 100))
            val_loss_list.append(val_loss)
            val_acc_list.append(val_acc)
            if val_acc > best_val_acc:
                accelerator.wait_for_everyone()
                log.debug("Deep copying new best model." +
                          "(validation of {:.4f}%, over {:.4f}%)".format(
                              val_acc * 100, best_val_acc * 100))
                if self.use_ema:
                    unwrapped_ema_model = accelerator.unwrap_model(self.ema_model.ema)
                    best_ema_model_to_save = deepcopy(unwrapped_ema_model.state_dict())
                    if self.save_dir:
                        accelerator.save(best_ema_model_to_save, self.save_dir + '/ema_model.pth.tar')

                unwrapped_model = accelerator.unwrap_model(self.model)
                best_model_to_save = deepcopy(unwrapped_model.state_dict())
                best_val_acc = val_acc
                if self.save_dir:
                    accelerator.save(best_model_to_save, self.save_dir + '/model.pth.tar')

            if self.opt_type == Optimizer.ADAM:
                self.lr_scheduler.step()

        accelerator.wait_for_everyone()
        self.model = accelerator.unwrap_model(self.model)
        if self.use_ema:
            self.ema_model.ema = accelerator.unwrap_model(self.ema_model.ema)
        
        if self.save_dir and accelerator.is_local_main_process:
            val_dic = {'train': train_loss_list, 'validation': val_loss_list}
            self.save_plot('loss', val_dic, self.save_dir)
            val_dic = {'train': train_acc_list, 'validation': val_acc_list}
            self.save_plot('accuracy', val_dic, self.save_dir)
        if self.select_on_val and best_model_to_save is not None:
            if self.use_ema and best_ema_model_to_save is not None:
                self.ema_model.ema.load_state_dict(best_ema_model_to_save)
            self.model.load_state_dict(best_model_to_save)
        if unlabeled_data is not None:
            unlabeled_data.transform = self.org_unlabeled_transform

    def _get_pred_classifier(self):
        return self.ema_model.ema if self.use_ema else self.model

    def _train_epoch(self, train_data_loader, unlabeled_data_loader=None):
        self.model.train()

        labeled_iter = iter(train_data_loader)
        if not self.use_scads:
            unlabeled_iter = iter(unlabeled_data_loader)

        running_loss = 0.0
        running_acc = 0.0
        acc_count = 0
        for i in range(self.steps_per_epoch):
            try:
                inputs_x, targets_x = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(train_data_loader)
                inputs_x, targets_x = next(labeled_iter)

            if not self.use_scads:
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
            if self.use_scads:
                inputs = inputs_x
            else:
                inputs = torch.cat((inputs_x, inputs_u_w, inputs_u_s))

            labels = targets_x

            self.optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                logits = self.model(inputs)

                if self.use_scads:
                    logits_x = logits
                    loss = F.cross_entropy(logits, labels, reduction='mean')
                else:
                    logits_x = logits[:batch_size]
                    logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
                    del logits

                    lx = F.cross_entropy(logits_x, labels, reduction='mean')
                    pseudo_label = torch.softmax(logits_u_w.detach() / self.temp, dim=-1)
                    max_probs, targets_u = torch.max(pseudo_label, dim=-1)
                    # binary mask to ignore unconfident psudolabels
                    mask = max_probs.ge(self.conf_thresh).float()

                    lu = (F.cross_entropy(logits_u_s, targets_u, reduction='none') * mask).mean()
                    loss = lx + self.lambda_u * lu

                accelerator.backward(loss)
                self.optimizer.step()

            if self.opt_type == Optimizer.SGD:
                self.lr_scheduler.step()
            if self.use_ema:
                self.ema_model.update(self.model)

            logits_x = accelerator.gather(logits_x.detach())
            labels = accelerator.gather(labels)

            running_loss += loss.item()
            running_acc += self._get_train_acc(logits_x, labels)
            acc_count += len(labels)

        epoch_loss = running_loss / self.steps_per_epoch
        epoch_acc = running_acc.item() / acc_count
        return epoch_loss, epoch_acc

    def _validate_epoch(self, val_data_loader):
        """
        Validate for one epoch.
        :param val_data_loader: A dataloader containing validation data
        :param use_gpu: Whether or not to use the GPU
        :return: None
        """
        eval_model = self.ema_model.ema if self.use_ema else self.model

        running_loss = 0
        running_acc = 0
        for batch in val_data_loader:
            inputs = batch[0]
            labels = batch[1]
            with torch.set_grad_enabled(False):
                outputs = eval_model(inputs)
                loss = torch.nn.functional.cross_entropy(outputs, labels)
                _, preds = torch.max(outputs, 1)

            preds = accelerator.gather(preds.detach())
            labels = accelerator.gather(labels)

            running_loss += loss.item()
            running_acc += torch.sum(preds == labels)

        epoch_loss = running_loss / len(val_data_loader.dataset)
        epoch_acc = running_acc.item() / len(val_data_loader.dataset)
        return epoch_loss, epoch_acc
