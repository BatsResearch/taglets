import torch
import torch.nn as nn
import numpy as np


# samples data in an episodic manner
class CategoriesSampler:
    def __init__(self, label, n_batch, n_cls, n_per):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per

        label = np.array(label)
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
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
