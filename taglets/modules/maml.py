from fewshot_utils import get_label_distr, validate_few_shot_config
from .module import Module
from ..pipeline import Taglet

import os
import torch
import torch.nn as nn
import torch.functional as F
import torch.multiprocessing as mp

import logging

log = logging.getLogger(__name__)


class Linear_fw(nn.Linear): # used in MAML to forward input with fast weight
    def __init__(self, in_features, out_features):
        super(Linear_fw, self).__init__(in_features, out_features)
        self.weight.fast = None #Lazy hack to add fast weight link
        self.bias.fast = None

    def forward(self, x):
        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.linear(x, self.weight.fast, self.bias.fast)
        else:
            out = super(Linear_fw, self).forward(x)
        return out


class MAMLModule(Module):
    def __init__(self, task):
        super().__init__(task)
        self.taglets = [MAMLTaglet(task, train_shot=5, query=15, episodes=20, use_scads=False)]


class MAMLTaglet(Taglet):
    def __init__(self, task, train_shot, query, episodes, approx=True, n_meta_tasks=4,
                                                                       val_shot=None,
                                                                       use_scads=True):
        super().__init__(task)

        self.name = 'maml'
        self.episodes = episodes

        self.train_shot = train_shot
        self.query = query

        if val_shot is None:
            self.val_shot = self.train_shot

        self.save_dir = os.path.join('trained_models', self.name)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.task = task
        self.train_shot = train_shot

        self.way = len(task.classes)
        self.query = query
        self.episodes = episodes

        if n_meta_tasks < 1:
            raise ValueError('Invalid number of meta-tasks: %d' % n_meta_tasks)
        self.n_meta_tasks = n_meta_tasks

        # set up model
        encoded_shape = self._get_model_output_shape(task.input_shape, self.model)
        classifier = Linear_fw(encoded_shape, self.way)
        classifier.bias.data.fill_(0)

        # whether to use first order linear approximation
        self.approx = approx

        self.model = nn.Sequential(self.model, classifier)
        self.use_scads = use_scads

        # Parameters needed to be updated based on freezing layer
        params_to_update = []
        for param in self.model.parameters():
            if param.requires_grad:
                params_to_update.append(param)

        self._params_to_update = params_to_update
        self.optimizer = torch.optim.Adam(self._params_to_update, lr=self.lr, weight_decay=1e-4)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.1)

        self.abstain = False

    def train(self, train_data, val_data, use_gpu):
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '9002'

        if len(train_data) == 0:
            log.debug('train dataset is empty! abstaining from labeling.')
            self.abstain = True
            return

        if self.use_scads:
            # merge val_data with train_data to form infer_data
            # Imagenet 1K is used as a base dataset and partitioned in train / val
            pass
        else:
            pass
            # use training data to build class prototypes

        # validate that train / val datasets are sufficiently large given shot / way
        train_label_distr = get_label_distr(train_data.labels)
        if not validate_few_shot_config('Train', train_label_distr, shot=self.train_shot,
                                        way=self.way, query=self.query):
            self.abstain = True
            return

        if val_data is not None:
            val_label_distr = get_label_distr(val_data.labels)

            if not validate_few_shot_config('Val', val_label_distr, shot=self.val_shot,
                                            way=self.way, query=self.query):
                val_data = None

        args = (self, train_data, val_data, use_gpu, self.n_proc)
        mp.spawn(self._do_train, nprocs=self.n_proc, args=args)

    def _set_forward(self, x):
        p = self.train_shot * self.way
        scores = self.model(x[:p])
        labels = torch.arange(self.way).repeat(self.train_shot)

        set_loss = self.criterion(scores, labels)

        fast_parameters = list(self.model.parameters())
        for weight in self.model.parameters():
            weight.fast = None

        self.model.zero_grad()

        grad = torch.autograd.grad(set_loss, fast_parameters, create_graph=True)
        if self.approx:
            grad = [g.detach() for g in grad]

        fast_parameters = []
        for k, weight in enumerate(self.model.parameters()):
            if weight.fast is None:
                weight.fast = weight - self.lr * grad[k]
            else:
                weight.fast = weight.fast - self.lr * grad[k]
            fast_parameters.append(weight.fast)

        return self.model(x[p:])

    def _train_epoch(self, train_data_loader, use_gpu):
        """
        Train for one epoch.
        :param train_data_loader: A dataloader containing training data
        :param cuda: Whether or not to use the GPU
        :return: None
        """
        self.model.train()

        if use_gpu and torch.cuda.is_available():
            self.model.cuda()

        running_loss = 0.0
        running_acc = 0.0
        count = 0
        avg_loss = 0
        acc = 0
        task_count = 0
        loss_all = []
        for i, batch in enumerate(train_data_loader, 1):
            log.info('Train Episode: %d' % i)
            count += 1
            if use_gpu:
                data, _ = [_.cuda() for _ in batch]
            else:
                data = batch[0]

            self.optimizer.zero_grad()

            labels = torch.arange(self.way).repeat(self.train_shot)
            query_scores = self._set_forward(data)
            loss = self.criterion(query_scores, labels)

            avg_loss = avg_loss + loss.data[0]
            loss_all.append(loss)

            task_count += 1

            if task_count == self.n_meta_tasks:
                loss_q = torch.stack(loss_all).sum(0)
                loss_q.backward()

                self.optimizer.step()
                task_count = 0
                loss_all = []

            running_loss += loss.item()
            running_acc += acc
            log.info("avg train episode loss: %f" % (loss.item() / self.query))
            log.info("train episode accuracy: %f%s" % (acc * 100.0, "%"))
        epoch_loss = running_loss / count if count > 0 else 0.0
        epoch_acc = running_acc / count if count > 0 else 0.0
        return epoch_loss, epoch_acc

    def _validate_epoch(self, val_data_loader, use_gpu):
        self.model.eval()

        if use_gpu and torch.cuda.is_available():
            self.model.cuda()

        running_loss = 0.0
        running_acc = 0.0
        count = 0
        for i, batch in enumerate(val_data_loader, 1):
            log.info('Val Episode: %d' % i)
            count += 1
            if use_gpu:
                data, _ = [_.cuda() for _ in batch]
            else:
                data = batch[0]

            self.optimizer.zero_grad()
            loss, acc = self.model.get_forward_loss(data,
                                                       use_gpu,
                                                       way=self.way,
                                                       shot=self.val_shot,
                                                       val=True)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            running_acc += acc
            log.info("avg val episode loss: %f" % (loss.item() / self.query))
            log.info("val episode accuracy: %f%s" % (acc * 100.0, "%"))
        epoch_loss = running_loss / count if count > 0 else 0.0
        epoch_acc = running_acc / count if count > 0 else 0.0
        return epoch_loss, epoch_acc