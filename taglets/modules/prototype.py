from .module import Module
from ..pipeline import Taglet
import copy
import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.autograd import Variable



from .fewshot_utils import DistributedCategoriesSampler, get_label_distr, \
    get_label_distr_stats, validate_few_shot_config, DistributedBatchCategoriesSampler
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)


def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    return ((a - b) ** 2).sum(dim=2)

def euclidean_dist(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)



def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    if torch.cuda.is_available():
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    else:
        return (pred == label).type(torch.FloatTensor).mean().item()


class Flatten(nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)


# this is kind of a hack
class NearestProtoModule(nn.Module):
    #TODO: determine whether prototypes will be passed by reference or value
    def __init__(self, prototypes, n_classes):
        self.prototypes = prototypes
        self.n_classes = n_classes
        super(NearestProtoModule, self).__init__()

    def forward(self, x):
        if self.training:
            return x
        else:
            batch_size = x.shape[0]
            # +1 for abstaining
            labels = torch.zeros((batch_size, self.n_classes + 1))
            for i in range(batch_size):
                # TODO: ensure x[i] is a vector
                label = PrototypeTaglet.onn(x[i], prototypes=self.prototypes)
                labels[label, label] = 1
            return labels


class PrototypeModule(Module):
    def __init__(self, task):
        super().__init__(task)
        episodes = 200 if not os.environ.get("CI") else 5
        self.taglets = [PrototypeTaglet(task, train_shot=5, train_way=30,
                                        query=15, episodes=episodes, use_scads=False)]


class PrototypeTaglet(Taglet):
    def __init__(self, task, train_shot, train_way, query, episodes, val_shot=None, val_way=None, use_scads=True):
        super().__init__(task)
        self.name = 'prototype'
        self.episodes = episodes
        self.prototypes = {}
        self.model = self.model
                              #     NearestProtoModule(self.prototypes, len(task.classes)))

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

        # Parameters needed to be updated based on freezing layer
        params_to_update = []
        for param in self.model.parameters():
            if param.requires_grad:
                params_to_update.append(param)

        self._params_to_update = params_to_update
        self.optimizer = torch.optim.Adam(self._params_to_update, lr=self.lr, weight_decay=1e-4)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.1)

        self.use_scads = use_scads
        self.abstain = False

    @staticmethod
    def onn(i, prototypes):
        min_dist = None
        lab = None
        for key, proto in prototypes.items():
            rel_dist = euclidean_metric(proto, i)
            if min_dist is None or rel_dist < min_dist:
                min_dist = rel_dist
                lab = key + 1

        if min_dist is None:
            log.warning('No prototypes found! Abstaining from labeling.')
            lab = 0
        return lab

    def _get_train_sampler(self, data, n_proc, rank):
        return DistributedBatchCategoriesSampler(rank=rank,
                                                labels=data.labels,
                                                n_episodes=self.episodes,
                                                n_cls=self.train_way,
                                                n_per=self.train_shot + self.query)

    def _get_val_sampler(self, data, n_proc, rank):
        return DistributedBatchCategoriesSampler(rank=rank,
                                                labels=data.labels,
                                                n_episodes=self.episodes,
                                                n_cls=self.val_way,
                                                n_per=self.val_shot + self.query)

    def _get_dataloader(self, data, sampler):
        return torch.utils.data.DataLoader(
            dataset=data, batch_sampler=sampler,
            num_workers=0, pin_memory=True)

    def build_prototypes(self, infer_data, use_gpu):
        self.model.eval()

        infer_dataloader = DataLoader(dataset=infer_data, batch_size=self.batch_size,
                                      num_workers=0, pin_memory=True)

        device = torch.device('cuda:0' if torch.cuda.is_available() and use_gpu else 'cpu')
        self.model.to(device)

        # TODO: potentially speed this up
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
                    try:
                        self.prototypes[lbl].append(proto)
                    except:
                        self.prototypes[lbl] = [proto]

            # form centroids
            for key, values in self.prototypes.items():
                self.prototypes[key] = torch.stack(values).mean(dim=0)

    def train(self, train_data, val_data, use_gpu):
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '9000'

        if len(train_data) == 0:
            log.debug('train dataset is empty! abstaining from labeling.')
            self.abstain = True

        infer_data = None
        if self.use_scads:
            # merge val_data with train_data to form infer_data
            # Imagenet 1K is used as a base dataset and partitioned in train / val
            pass
        else:
            # use training data to build class prototypes
            infer_data = train_data

        # validate that train / val datasets are sufficiently large given shot / way
        train_label_distr = get_label_distr(train_data.labels)
        if not validate_few_shot_config('Train', train_label_distr, shot=self.train_shot,
                                        way=self.train_way, query=self.query):
            self.abstain = True
            return

        if val_data is not None:
            val_label_distr = get_label_distr(val_data.labels)

            if not validate_few_shot_config('Val', val_label_distr, shot=self.val_shot,
                                            way=self.val_way, query=self.query):
                self.abstain = True
                val_data = None

        args = (self, train_data, val_data, use_gpu, self.n_proc)
        #mp.spawn(self._do_train, nprocs=self.n_proc, args=args)


        sampler = self._get_train_sampler(train_data, 1, 0)
        self._train_epoch(self._get_dataloader(train_data, sampler), False)

        # after child training processes are done (waitpid!), construct prototypes
        self.build_prototypes(infer_data, use_gpu=use_gpu)

    def _train_epoch(self, train_data_loader, use_gpu):
        """
        Train for one epoch.
        :param train_data_loader: A dataloader containing training data
        :param cuda: Whether or not to use the GPU
        :return: None
        """
        self.model.train()
        use_gpu = use_gpu and torch.cuda.is_available()

        #label = torch.arange(self.train_way).repeat(self.query)
        #if use_gpu:
        #    label = label.type(torch.cuda.LongTensor)
        #else:
        #    label = label.type(torch.LongTensor)

        running_loss = 0.0
        count = 0
        for i, batch in enumerate(train_data_loader, 1):
            count += 1
            if torch.cuda.is_available():
                data, _ = [_.cuda() for _ in batch]
            else:
                data = batch[0]

            #with torch.autograd.set_detect_anomaly(True):
                with torch.autograd.set_detect_anomaly(True):
                    p = self.train_way * self.train_shot
                    data_shot, data_query = data[:p], data[p:]


                    # calculate barycenter for each class
                    proto = self.model(data_shot)
                    proto = proto.reshape(self.train_shot, self.train_way, -1).mean(dim=0)

                    label = torch.from_numpy(np.repeat(range(self.train_way), self.query))
                    label = Variable(label)
                       # if use_gpu:
                       #     label = label.cuda()

                    query = self.model(data_query).reshape(self.train_way * self.query, -1)
                    logits = -euclidean_metric(query, proto)
                    loss = self.criterion(logits, label)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            running_loss += loss.item()
        epoch_loss = running_loss / count
        return epoch_loss

    def _validate_epoch(self, val_data_loader, use_gpu):
        self.model.eval()
        label = torch.arange(self.val_way).repeat(self.query)
        use_gpu = use_gpu and torch.cuda.is_available()

        if use_gpu:
            label = label.type(torch.cuda.LongTensor)
        else:
            label = label.type(torch.LongTensor)

        running_loss = 0.0
        running_acc = 0.0
        count = 0
        for i, batch in enumerate(val_data_loader, 1):
            count += 1
            if torch.cuda.is_available():
                data, _ = [_.cuda() for _ in batch]
            else:
                data = batch[0]

            p = self.val_shot * self.val_way
            data_shot, data_query = data[:p], data[p:]

            # calculate barycenter for each class
            proto = self.model(data_shot).reshape(self.val_shot, self.val_way, -1).mean(dim=0)
            query_proto = self.model(data_query)

            logits = euclidean_metric(query_proto, proto)
            loss = F.cross_entropy(logits, label)

            running_loss += loss.item()
            running_acc += count_acc(logits, label)

        epoch_loss = running_loss / count
        epoch_acc = running_acc / count
        return epoch_loss, epoch_acc
