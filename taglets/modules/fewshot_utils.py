import torch
import numpy as np
import logging

from ..pipeline import Taglet
from torch.utils.data import Sampler

log = logging.getLogger(__name__)

class MetaTaglet(Taglet):
    def __init__(self, task, train_shot, train_way, query, episodes, val_shot, val_way, use_scads):
        super().__init__(task)
        self.task = task
        self.train_shot = train_shot
        self.train_way = train_way

        self.query = query
        self.episodes = episodes

        self.val_shot = val_shot
        self.val_way = val_way
        self.use_scads = use_scads

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
        # set a large, deterministic seed
        g.manual_seed((rank + 1) * (1 << 15))
        super(DistributedBatchCategoriesSampler, self).__init__(labels=labels,
                                                               n_episodes=n_episodes,
                                                               n_cls=n_cls,
                                                               n_per=n_per,
                                                               rand_generator=g)

    def set_epoch(self, epoch):
        pass


class DistributedCategoriesSampler(CategoriesSampler):
    def __init__(self, rank, labels, n_cls, n_per):
        g = torch.Generator()
        # set a large, deterministic seed
        g.manual_seed((rank + 1) * (1 << 15))
        super(DistributedCategoriesSampler, self).__init__(labels=labels,
                                                           n_cls=n_cls,
                                                           n_per=n_per,
                                                           rand_generator=g)

    def set_epoch(self, epoch):
        pass


def get_label_distr(labels):
    distr = {}

    for label in labels:
        l = label.item()
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
