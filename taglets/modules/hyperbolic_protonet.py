from .module import Module
from ..pipeline import Taglet
import copy
import os
import logging
import torch
import torch.nn as nn
import numpy as np

import torch.nn.functional as F
from torch.utils.data import DataLoader
from ..hyptorch.nn import ToPoincare
from ..hyptorch.pmath import poincare_mean, dist_matrix, dist


log = logging.getLogger(__name__)


class HyperbolicProtoModule(Module):
    def __init__(self, task):
        super().__init__(task)
        self.taglets = [HyperbolicProtoTaglet(task, way=task.classes)]


# adapted from https://arxiv.org/pdf/1904.02239.pdf and
# https://github.com/leymir/hyperbolic-image-embeddings
class HyperbolicProtoTaglet(Taglet):
    # c represents the curvature of the poincare ball; use either 0.05 or 0.01?
    # train_c not set in code
    def __init__(self, task, shot=None, way=None, query=5, c=0.01, gamma=0.5,
                 lr=0.001, step=40, train_c=False, train_x=False, n_batch=100):
        super().__init__(task)
        self.name = 'hyperbolic-protonet'
        self.classes = len(task.classes.keys())
        self.train_label_distr = task.get_train_label_distr()
        self.validation_label_distr = task.get_validation_label_distr()

        # labels for all examples in training and validation datasets
        self.train_labels = task.get_train_labels()
        self.val_labels = task.get_validation_labels()

        assert query is not None and 1 <= query
        self.query = query

        # way and shot will be None if invalid
        self.shot, self.way, self.train_degenerate_labels = \
            self._determine_fewshot_params(shot, way, self.train_label_distr)

        self.n_valid_labels = self.classes - len(self.train_degenerate_labels.keys())

        self.val_shot, self.val_way, self.val_degenerate_labels = \
            self._determine_fewshot_params(self.shot, self.n_valid_labels, self.validation_label_distr)

        if self.val_shot != self.shot:
            print('Warning :: HyperbolicProtoTaglet -> train shot not equal to val shot.')
            print('train shot: ' + str(self.shot))
            print('val shot: ' + str(self.val_shot))

        if self.shot is None or self.way is None:
            print('Warning :: HyperbolicProtoTaglet -> way or shot is zero. '
                  'Consider reducing query.')

        # set training hyperparameters
        self.gamma = gamma
        self.lr = lr
        self.step = step
        # for episodic training
        self.n_batch = n_batch

        # remove fully connected layers but keep encoder; TODO: ensure this works
        #self.model.fc = torch.nn.Identity()

        # euclidean to poincare layer
        hyper_encoder = HyperbolicEncoder(euclid_encoder=self.model,
                                                                 c=c,
                                                                 train_c=train_c,
                                                                 train_x=train_x)
        self.model = nn.Sequential(self.model, hyper_encoder)
        self.c = hyper_encoder.e2p.c

        self.save_dir = os.path.join('trained_models', self.name)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # Parameters needed to be updated based on freezing layer
        params_to_update = []
        for param in self.model.parameters():
            if param.requires_grad:
                params_to_update.append(param)

        self._params_to_update = params_to_update
        self.optimizer = torch.optim.Adam(self._params_to_update, lr=self.lr, weight_decay=1e-4)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.step,
                                                            gamma=self.gamma)

    def _determine_fewshot_params(self, desired_shot, desired_way, label_distr):

        # labels without enough training examples; abstain for these
        degenerate_labels = {}
        supp_shot = None
        i = 0
        for label in degenerate_labels:
            support = label_distr[label] - self.query
            if support <= 0:
                degenerate_labels[label] = 1
                i += 1
            else:
                if supp_shot is None:
                    supp_shot = support
                supp_shot = min(supp_shot, support)

        # kinda convoluted
        if supp_shot is None:
            desired_shot = None
        else:
            desired_shot = supp_shot if desired_shot is None else min(supp_shot, desired_shot)

        if desired_way is None:
            desired_way = self.classes
        desired_way = desired_way - i
        if desired_way == 0:
            desired_way = None
        return desired_shot, desired_way, degenerate_labels

    def _filter_labeled_data(self, labels, degenerate_labels):
        # returns indices of examples with valid labels and num of valid labels
        valid_indices = []
        for i in range(labels, degenerate_labels):
            if labels[i] not in degenerate_labels:
                valid_indices.append(i)
        return valid_indices

    def train(self, train_data_loader, val_data_loader, use_gpu):
        # in the case where training set is insufficient for learning
        if self.way is None or self.shot is None:
            return

        # TODO: GPU memory safety
        self.model = self.model.cuda() if use_gpu else self.model.cpu()

        # Next, we'll set up a proper training data loader
        valid_train_indices = self._filter_labeled_data(self.train_labels, self.train_degenerate_labels)
        valid_train_labels = [x for x in range(1, self.classes) if x not in self.train_degenerate_labels]
        train_sampler = CategoriesSampler(valid_train_indices, valid_train_labels,
                                          lambda l: np.array([self.train_labels[idx] for idx in l]),
                                          n_batch=self.n_batch, n_cls=self.way,
                                          n_per=self.shot * self.query)

        batch_train_data_loader = DataLoader(train_data_loader.dataset,
                                       batch_sampler=train_sampler,
                                       num_workers=8,
                                       pin_memory=True)

        sufficient_val_data = self.val_way is not None and self.val_shot is not None

        if sufficient_val_data:
            valid_val_indices = self._filter_labeled_data(self.val_labels, self.val_degenerate_labels)
            valid_val_labels = [x for x in range(1, self.classes) if x not in self.val_degenerate_labels]
            val_sampler = CategoriesSampler(valid_val_indices, valid_val_labels,
                                            lambda l: np.array([self.val_labels[idx] for idx in l]),
                                            n_batch=self.n_batch, n_cls=self.val_way,
                                            n_per=self.val_shot * self.query)

            batch_val_data_loader = DataLoader(val_data_loader,
                                           batch_sampler=val_sampler,
                                           num_workers=8,
                                           pin_memory=True)
        else:
            print('Warning :: HyperbolicProtoTaglet -> insufficient amount of validation data. Training '
                  'will still occur but validation will not.')

        best_model_to_save = None
        for epoch in range(self.num_epochs):
            log.info('epoch: {}'.format(epoch))

            # Train on training data
            train_loss = self._train_epoch(batch_train_data_loader, use_gpu)
            log.info('train loss: {:.4f}'.format(train_loss))

            # Evaluation on validation data
            if not val_data_loader:
                val_loss = 0
                val_acc = 0
                continue

            val_loss, val_acc = self._validate_epoch(batch_val_data_loader, use_gpu)
            log.info('validation loss: {:.4f}'.format(val_loss))
            log.info('validation acc: {:.4f}%'.format(val_acc*100))

            if self.lr_scheduler:
                self.lr_scheduler.step()

            if val_acc > self._best_val_acc:
                log.info("Deep copying new best model." +
                         "(validation of {:.4f}%, over {:.4f}%)".format(val_acc, self._best_val_acc))
                self._best_val_acc = val_acc
                best_model_to_save = copy.deepcopy(self.model.state_dict())
                torch.save(best_model_to_save, self.save_dir + '/model.pth.tar')

        log.info("Epoch {} result: ".format(epoch + 1))
        log.info("Average training loss: {:.4f}".format(train_loss))
        log.info("Average validation loss: {:.4f}".format(val_loss))
        log.info("Average validation accuracy: {:.4f}%".format(val_acc * 100))

        if self.select_on_val and best_model_to_save:
            # Reloads best model weights
            self.model.load_state_dict(best_model_to_save)

        self.model.eval()
        self.prototypes = {}
        for data in train_data_loader:
            with torch.set_grad_enabled(False):
                image, label = data[0], data[1]

                # Memorize
                if use_gpu:
                    image = image.cuda()
                    label = label.cuda()

                for img, lbl in zip(image, label):
                    if int(lbl) in self.train_degenerate_labels:
                        continue
                    proto = self.model(torch.unsqueeze(img, dim=0))
                    lbl = int(lbl.item())

                    # TODO: determine whether this will work; also this will be very slow...
                    if lbl in self.prototypes:
                        self.prototypes[lbl] = poincare_mean(self.prototypes[lbl], proto)
                    else:
                        self.prototypes[lbl].append(proto)

    def _train_epoch(self, train_data_loader, use_gpu):
        """
        Train for one epoch.
        :param train_data_loader: A dataloader containing training data
        :param cuda: Whether or not to use the GPU
        :return: None
        """
        self.model.train()

        label = torch.arange(self.way).repeat(self.query)
        use_gpu = use_gpu and torch.cuda.is_available()

        if use_gpu:
            label = label.type(torch.cuda.LongTensor)
        else:
            label = label.type(torch.LongTensor)

        running_loss = 0.0
        count = 0
        for i, batch in enumerate(train_data_loader, 1):
            count += 1
            if torch.cuda.is_available():
                data, _ = [_.cuda() for _ in batch]
            else:
                data = batch[0]

            p = self.shot * self.way
            data_shot, data_query = data[:p], data[p:]

            with torch.set_grad_enabled(True):
                proto = self.model(data_shot).reshape(self.shot, self.way, -1)
                proto = poincare_mean(proto, dim=0, c=self.c)
                logits = (-dist_matrix(self.model(data_query), proto, c=self.c))
                loss = F.cross_entropy(logits, label)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            running_loss += loss.item()
        epoch_loss = running_loss / len(count)
        return epoch_loss

    def _validate_epoch(self, val_data_loader, use_gpu):
        """
        Validate for one epoch.
        :param val_data_loader: A dataloader containing validation data
        :return: None
        """
        self.model.eval()

        label = torch.arange(self.val_way).repeat(self.query)
        use_gpu = use_gpu and torch.cuda.is_available()

        if use_gpu:
            label = label.type(torch.cuda.LongTensor)
        else:
            label = label.type(torch.LongTensor)

        running_loss = 0.0
        running_acc = 0.0
        for i, batch in enumerate(val_data_loader, 1):
            if torch.cuda.is_available():
                data, _ = [_.cuda() for _ in batch]
            else:
                data = batch[0]

            p = self.val_shot * self.val_way
            data_shot, data_query = data[:p], data[p:]

            with torch.set_grad_enabled(True):
                proto = self.model(data_shot).reshape(self.shot, self.way, -1)
                proto = poincare_mean(proto, dim=0, c=self.c)
                logits = (-dist_matrix(self.model(data_query), proto, c=self.c))
                loss = F.cross_entropy(logits, label)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            running_loss += loss.item()
            running_acc += count_acc(logits, label)

        epoch_loss = running_loss / len(val_data_loader.dataset)
        epoch_acc = running_acc / len(val_data_loader.dataset)
        return epoch_loss, epoch_acc

    def nn(self, data):
        min_dist = float("Inf")
        lab = None

        for key in self.prototypes:
            rel_dist = dist(key, data, c=self.c)
            if rel_dist < min_dist:
                min_dist = rel_dist
                lab = key
        return lab

    def execute(self, unlabeled_data_loader, use_gpu):
        self.model.eval()

        abstain_all = self.way is None or self.shot is None

        if abstain_all:
            print('Warning :: HyperbolicProtoTaglet -> abstaining for unlabeled examples')

        use_gpu = use_gpu and abstain_all

        # don't waste GPU memory on abstaining
        self.model = self.model.cuda() if use_gpu else self.model.cpu()

        # TODO: make abstaining more memory efficient
        predicted_labels = []
        for inputs in unlabeled_data_loader:
            assert inputs.ndim == 4
            if abstain_all:
                predicted_labels.append([0] * inputs.shape[0])
            else:
                if use_gpu:
                    inputs = inputs.cuda()
                with torch.set_grad_enabled(False):
                    for data in inputs:
                        data = torch.unsqueeze(data, dim=0)
                        proto = self.model(data)
                        prediction = self.nn(proto)
                        predicted_labels.append(prediction)
        return predicted_labels


def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    if torch.cuda.is_available():
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    else:
        return (pred == label).type(torch.FloatTensor).mean().item()


class HyperbolicEncoder(nn.Module):
    def __init__(self, euclid_encoder, c, train_c, train_x):
        super().__init__()
        self.encoder = euclid_encoder
        self.e2p = ToPoincare(c=c, train_c=train_c, train_x=train_x)

    def forward(self, data_shot, data_query):
        return self.e2p(self.encoder(data_shot))


class CategoriesSampler:
    def __init__(self, target_indices, valid_labels, idx_to_target, n_batch, n_cls, n_per):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per

        self.m_ind = []

        targets = np.array(target_indices)
        for label in valid_labels:
            label_indices = np.argwhere(label == idx_to_target(targets))
            ind = targets[label_indices]
            ind = torch.from_numpy(ind.reshape(-1))
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            classes = torch.randperm(len(self.m_ind))[: self.n_cls]
            for c in classes:
                l = self.m_ind[c]
                pos = torch.randperm(len(l))[: self.n_per]
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            yield batch
