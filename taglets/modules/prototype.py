from .module import Module
from ..pipeline import Taglet
import os
import logging
import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader, Sampler

log = logging.getLogger(__name__)


# TODO: put these samplers in a separate file
class CategoriesSampler(Sampler):
    def __init__(self, labels, n_cls, n_per, rand_generator=None):
        super().__init__(labels)
        # number of classes in dataset
        self.n_cls = n_cls
        # number of examples per class to be extracted
        self.n_per = n_per
        self.rand_generator = rand_generator

        self.m_ind = []
        labels = np.array(labels)
        for i in range(max(labels) + 1):
            # indices where labels are the same
            ind = np.argwhere(labels == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_cls * self.n_per

    def __iter__(self):
        batch = []
        if self.rand_generator:
            classes = torch.randperm(len(self.m_ind), generator=self.rand_generator)
        else:
            classes = torch.randperm(len(self.m_ind))

        classes = classes[: self.n_cls]
        for c in classes:
            l = self.m_ind[c]
            if self.rand_generator:
                pos = torch.randperm(len(l), generator=self.rand_generator)
            else:
                pos = torch.randperm(len(l))
            pos = pos[: self.n_per]
            batch.append(l[pos])
        return iter(batch)


# samples data in an episodic manner
class BatchCategoriesSampler(CategoriesSampler):
    def __init__(self, labels, n_episodes, n_cls, n_per, rand_generator):
        super().__init__(labels, n_cls, n_per, rand_generator)
        self.n_episodes = n_episodes

    def __iter__(self):
        for i_batch in range(self.n_episodes):
            batch = []

            if self.rand_generator:
                classes = torch.randperm(len(self.m_ind), generator=self.rand_generator)
            else:
                classes = torch.randperm(len(self.m_ind))
            classes = classes[: self.n_cls]

            for c in classes:
                l = self.m_ind[c]
                if self.rand_generator:
                    pos = torch.randperm(len(l), generator=self.rand_generator)
                else:
                    pos = torch.randperm(len(l))
                pos = pos[: self.n_per]
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            yield batch


class DistributedBatchCategoriesSampler(BatchCategoriesSampler):
    def __init__(self, rank, labels, n_episodes, n_cls, n_per):
        g = torch.Generator()
        # set a large, deterministic seed so that different processes get different data
        g.manual_seed((rank + 1) * (1 << 15))
        super(DistributedBatchCategoriesSampler, self).__init__(labels=labels,
                                                               n_episodes=n_episodes,
                                                               n_cls=n_cls,
                                                               n_per=n_per,
                                                               rand_generator=g)

    def set_epoch(self, epoch):
        pass


def get_label_distr(labels):
    distr = {}

    for label in labels:
        try:
            l = label.item()
        except AttributeError:
            l = label
        if l not in distr:
            distr[l] = 0
        distr[l] += 1
    return distr


def get_label_distr_stats(distr):
    max_label = None
    min_label = None

    for label in distr:
        density = distr[label]
        if max_label is None or density > max_label:
            max_label = density
        if min_label is None or density < min_label:
            min_label = density
    return max_label, min_label


def get_dataset_labels(dataset):
    labels = []
    log.warning('Manually getting dataset labels. This might cause a Memory Overflow.')
    for _, label in dataset:
        labels.append(label)
    return labels


def validate_few_shot_config(dataset_name, data_distr, shot, way, query):
    """
    validates that a dataset is sufficiently large for a given way / shot / query combination.

    :param dataset_name: Name of the dataset to validate
    :param data_distr: A label distribution returned from get_label_distr
    :param shot: number of examples per class
    :param way: number of classes per batch
    :param query: number of test examples per class
    return: whether the inputted few shot config is valid
    """
    if len(data_distr.keys()) < way:
        log.warning('%s dataset is too small for selected way (%d). Dataset contains %d classes'
                    % (dataset_name, way, len(data_distr.keys())))
        return False

    _, base_min_labels = get_label_distr_stats(data_distr)
    if base_min_labels < query + shot:
        log.warning('%s dataset is too small for selected shot (%d) and query (%d). '
                    'Smallest class contains %d point(s).' %
                    (dataset_name, shot, query, base_min_labels))
        return False
    return True


def calc_meta_params(distr, desired_way, desired_shot, desired_query):
    assert desired_way > 0 and desired_shot > 0
    num_classes = len(distr.keys())

    if num_classes == 0:
        raise ValueError('Label distribution has no data points. No meta params can be calculated.')
    way = num_classes if num_classes < desired_way else desired_way

    _, base_min_labels = get_label_distr_stats(distr)
    shot = min(base_min_labels, desired_shot)
    if base_min_labels - shot >= desired_query:
        return way, shot, desired_query

    if base_min_labels <= 1:
        raise ValueError('Label distribution is too small. No meta params can be calculated.')

    query = 0
    shot = 0
    idx = 0
    while 1:
        if base_min_labels == query + shot:
            break
        # distribute based on three query for every shot
        if idx % 3 == 0:
            shot += 1
        else:
            query += 1
        idx += 1

    if query - shot > 2:
        query -= 1
        shot += 1
    return way, shot, query


def count_acc(logits, label, use_gpu):
    pred = torch.argmax(logits, dim=1)
    if use_gpu:
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    else:
        return (pred == label).type(torch.FloatTensor).mean().item()


def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    return ((a - b) ** 2).sum(dim=2)


class NearestProtoModule(nn.Module):
    def __init__(self, prototypes, n_classes, encoder, shot, way, query):
        super(NearestProtoModule, self).__init__()
        self.prototypes = prototypes
        self.n_classes = n_classes
        self.encoder = encoder
        self.shot = shot
        self.way = way
        self.query = query
        self.criterion = nn.CrossEntropyLoss()
        self.val = False
        self.abstain = False

    # abstain from all labeling
    def set_label_abstaining(self, abstain):
        self.abstain = abstain

    def set_way(self, new_way):
        self.way = new_way

    def set_shot(self, new_shot):
        self.shot = new_shot

    def set_query(self, new_query):
        self.query = new_query

    def forward(self, x):
        embeddings = self.encoder(x)
        if self.training or self.val:
            if self.val:
                self.val = False

            # warning this will return an encoded vector rather than a label
            return embeddings
        else:
            batch_size = x.shape[0]
            # +1 for abstaining
            #TODO: add back abstaining
            #labels = torch.zeros((batch_size, self.n_classes + 1))
            labels = torch.zeros((batch_size, self.n_classes))
            if self.abstain:
                return labels
            for i in range(batch_size):
                label = PrototypeTaglet.onn(embeddings[i], prototypes=self.prototypes)
                labels[i, label] = 1
            return labels

    # hook for calculating forward
    def _get_forward(self, x, way=None, shot=None, val=False):
        if way is None:
            way = self.way

        if shot is None:
            shot = self.shot

        # way / shot stays constant while training
        assert (self.shot == shot and self.way == way) or val

        self.val = val

        p = self.shot * self.way
        encoding = self.forward(x)
        shot_proto = encoding[:p]
        query_proto = encoding[p:]

        # calculate barycenter for each class
        proto = shot_proto.reshape(shot, way, -1).mean(dim=0)
        return -euclidean_metric(query_proto, proto)

    def get_forward_loss(self, x, use_gpu, way=None, shot=None, val=False):
        if way is None:
            way = self.way

        if shot is None:
            shot = self.shot

        label = torch.arange(way).repeat(self.query)
        if use_gpu:
            label.cuda()

        logits = self._get_forward(x, way=way, shot=shot, val=val)
        return self.criterion(logits, label), count_acc(logits, label, use_gpu)


class PrototypeModule(Module):
    def __init__(self, task, auto_meta_param=False):
        super().__init__(task)
        episodes = 20 if not os.environ.get("CI") else 20
        self.taglets = [PrototypeTaglet(task, train_shot=5, train_way=10,
                                                            query=15,
                                                            episodes=episodes,
                                                            auto_meta_param=auto_meta_param,
                                                            use_scads=False)]


class PrototypeTaglet(Taglet):
    def __init__(self, task, train_shot, train_way, query,
                 episodes, auto_meta_param, val_shot=None, val_way=None, use_scads=True):
        super().__init__(task)
        self.name = 'prototype'
        self.episodes = episodes
        self.prototypes = {}

        # when testing, these parameters matter less
        self.train_shot = train_shot
        # ensure train_way is not more than num of classes in dataset
        self.train_way = min(len(task.classes), train_way)
        self.query = query

        if val_shot is None:
            self.val_shot = self.train_shot

        if val_way is None:
            self.val_way = self.train_way

        self.save_dir = os.path.join('trained_models', self.name)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.1)

        self.protonet = NearestProtoModule(self.prototypes,
                                        len(task.classes),
                                        encoder=self.model,
                                        shot=self.train_shot,
                                        way=self.train_way,
                                        query=self.query)
        self.use_scads = use_scads

        self.train_labels = None
        self.val_labels = None

        self.auto_meta_param = auto_meta_param

    @staticmethod
    def onn(i, prototypes):
        min_dist = None
        lab = None
        for key, proto in prototypes.items():
            rel_dist = torch.dist(proto, i)
            if min_dist is None or rel_dist < min_dist:
                min_dist = rel_dist
                lab = key
                #TODO: add back abstaining
                #lab = key + 1

        if min_dist is None:
            log.warning('No prototypes found! Abstaining from labeling.')
            lab = 0
        return lab

    def _get_train_sampler(self, data, n_proc, rank):
        return DistributedBatchCategoriesSampler(rank=rank,
                                                labels=self.train_labels,
                                                n_episodes=self.episodes,
                                                n_cls=self.train_way,
                                                n_per=self.train_shot + self.query)

    def _get_val_sampler(self, data, n_proc, rank):
        return DistributedBatchCategoriesSampler(rank=rank,
                                                labels=self.val_labels,
                                                n_episodes=self.episodes,
                                                n_cls=self.val_way,
                                                n_per=self.val_shot + self.query)

    def _get_dataloader(self, data, sampler):
        return torch.utils.data.DataLoader(
            dataset=data, batch_sampler=sampler,
            num_workers=0, pin_memory=True)

    def _get_pred_classifier(self):
        return self.protonet

    def build_prototypes(self, infer_data, use_gpu):
        self.protonet.eval()

        infer_dataloader = DataLoader(dataset=infer_data, batch_size=self.batch_size,
                                      num_workers=0, pin_memory=True)
        if use_gpu and torch.cuda.is_available():
            self.protonet.cuda()

        for data in infer_dataloader:
            with torch.set_grad_enabled(False):
                image, label = data[0], data[1]
                # Memorize
                if use_gpu:
                    image = image.cuda()
                    label = label.cuda()

                for img, lbl in zip(image, label):
                    proto = self.model(torch.unsqueeze(img, dim=0))
                    lbl = int(lbl.item())
                    if lbl not in self.prototypes:
                        self.prototypes[lbl] = []
                    self.prototypes[lbl].append(proto)

        # form centroids
        for key, values in self.prototypes.items():
            self.prototypes[key] = torch.stack(values).mean(dim=0)

    def train(self, train_data, val_data, use_gpu):
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '8888'

        if len(train_data) == 0:
            log.debug('train dataset is empty! abstaining from labeling.')
            self.protonet.set_label_abstaining(True)
            return

        infer_data = None
        if self.use_scads:
            # merge val_data with train_data to form infer_data
            # Imagenet 1K is used as a base dataset and partitioned in train / val
            pass
        else:
            # use training data to build class prototypes
            infer_data = train_data

        # validate that train / val datasets are sufficiently large given shot / way
        try:
            self.train_labels = train_data.labels
        except AttributeError:
            self.train_labels = get_dataset_labels(train_data)
        train_label_distr = get_label_distr(self.train_labels)

        if self.auto_meta_param:
            log.info('Automatically calculating meta parameters. Note that this might produce suboptimal results.')
            try:
                self.train_way, self.train_shot, self.query = calc_meta_params(train_label_distr,
                                                                               self.train_way,
                                                                               self.train_shot,
                                                                               self.query)
                self.protonet.set_way(self.train_way)
                self.protonet.set_shot(self.train_shot)
                self.protonet.set_query(self.query)
            except ValueError:
                self.protonet.set_label_abstaining(True)
                return
        elif not validate_few_shot_config('Train', train_label_distr, shot=self.train_shot,
                                        way=self.train_way, query=self.query):
            self.protonet.set_label_abstaining(True)
            return

        if val_data is not None:
            try:
                self.val_labels = val_data.labels
            except AttributeError:
                self.val_labels = get_dataset_labels(val_data)
            val_label_distr = get_label_distr(self.val_labels)

            if not validate_few_shot_config('Val', val_label_distr, shot=self.val_shot,
                                            way=self.val_way, query=self.query):
                val_data = None

        super().train(infer_data, val_data, use_gpu)
        self.build_prototypes(infer_data, use_gpu=use_gpu)

    def _train_epoch(self, rank, train_data_loader, use_gpu):
        """
        Train for one epoch.
        :param train_data_loader: A dataloader containing training data
        :param cuda: Whether or not to use the GPU
        :return: None
        """
        self.protonet.train()

        if use_gpu:
            self.protonet = self.protonet.cuda(rank)
        else:
            self.protonet = self.protonet.cpu()

        running_loss = 0.0
        running_acc = 0.0
        count = 0
        for i, batch in enumerate(train_data_loader, 1):
            log.info('Train Episode: %d' % i)
            count += 1
            if use_gpu:
                data, _ = [x.cuda(rank) for x in batch]
            else:
                data = batch[0]

            self.optimizer.zero_grad()
            loss, acc = self.protonet.get_forward_loss(data, use_gpu)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            running_acc += acc
            log.info("avg train episode loss: %f" % (loss.item() / self.query))
            log.info("train episode accuracy: %f%s" % (acc * 100.0, "%"))
        epoch_loss = running_loss / count if count > 0 else 0.0
        epoch_acc = running_acc / count if count > 0 else 0.0
        return epoch_loss, epoch_acc

    def _validate_epoch(self, rank, val_data_loader, use_gpu):
        self.protonet.eval()

        if use_gpu and torch.cuda.is_available():
            self.protonet = self.protonet.cuda(rank)
        else:
            self.protonet = self.protonet.cpu()

        running_loss = 0.0
        running_acc = 0.0
        count = 0
        for i, batch in enumerate(val_data_loader, 1):
            log.info('Val Episode: %d' % i)
            count += 1
            if use_gpu:
                data, _ = [x.cuda(rank) for x in batch]
            else:
                data = batch[0]
            with torch.set_grad_enabled(False):
                loss, acc = self.protonet.get_forward_loss(data,
                                                           use_gpu,
                                                           way=self.val_way,
                                                           shot=self.val_shot,
                                                           val=True)
            running_loss += loss.item()
            running_acc += acc
            log.info("avg val episode loss: %f" % (loss.item() / self.query))
            log.info("val episode accuracy: %f%s" % (acc * 100.0, "%"))
        epoch_loss = running_loss / count if count > 0 else 0.0
        epoch_acc = running_acc / count if count > 0 else 0.0
        return epoch_loss, epoch_acc

