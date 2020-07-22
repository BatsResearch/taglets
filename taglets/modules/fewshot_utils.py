import torch
import numpy as np
import logging
import math

from torch.utils.data import Sampler

log = logging.getLogger(__name__)


# samples data in an episodic manner
class CategoriesSampler(Sampler):
    def __init__(self, labels, n_episodes, n_cls, n_per):
        super().__init__(labels)
        self.n_episodes = n_episodes
        # number of classes in dataset
        self.n_cls = n_cls
        # number of examples per class to be extracted
        self.n_per = n_per

        labels = np.array(labels)
        self.m_ind = []
        for i in range(max(labels) + 1):
            ind = np.argwhere(labels == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i_batch in range(self.n_episodes):
            batch = []
            classes = torch.randperm(len(self.m_ind))[: self.n_cls]
            for c in classes:
                l = self.m_ind[c]
                pos = torch.randperm(len(l))[: self.n_per]
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            yield batch


class DistributedCategoriesSampler(CategoriesSampler):
    def __init__(self, num_replicas, labels, n_episodes, n_cls, n_per):
        n_episodes_per_proc = int(math.ceil(n_episodes * 1.0 / num_replicas))
        super(DistributedCategoriesSampler, self).__init__(labels=labels,
                                                           n_episodes=n_episodes_per_proc,
                                                           n_cls=n_cls,
                                                           n_per=n_per)

    # this is required for training, but we already perform shuffling automatically
    def set_epoch(self, epoch):
        pass


def get_label_distr(labels):
    distr = {}

    for label in labels:
        if label not in distr:
            distr[label] = 0
        distr[label] += 1
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
                    'Smallest class contains %d points.' %
                    (dataset_name, shot, query, base_min_labels))
        return False
    return True
