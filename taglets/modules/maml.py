from .module import Module
from ..pipeline import Taglet

import logging

log = logging.getLogger(__name__)


class MAMLModule(Module):
    def __init__(self, task):
        super().__init__(task)
        self.taglets = [MAMLTaglet(task, use_scads=False)]


class MAMLTaglet(Taglet):
    def __init__(self, task, use_scads=True):
        pass

    def train(self, train_data_loader, val_data_loader, use_gpu):
        pass

    def execute(self, unlabeled_data_loader, use_gpu):
        pass