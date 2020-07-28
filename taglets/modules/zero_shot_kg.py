from taglets.data.custom_dataset import CustomDataset
from torch.utils import data

from .module import Module
from ..pipeline import Taglet
from ..scads.interface.scads import Scads

import os
import re
import json
import random
import tempfile
import torch
import torch.nn as nn
import logging
import copy
import numpy as np
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision.models as models

from tqdm import tqdm
from taglets.modules.zsl_kg_lite import ZSLKG

log = logging.getLogger(__name__)


class ZSLKGModule(Module):
    def __init__(self, task):
        super().__init__(task)
        self.taglets = [ZSLKGTaglet(task)]


class ZSLKGTaglet(Taglet):
    def __init__(self, task):
        super().__init__(task)
        self.name = 'zsl-kg'
        root_path = Scads.get_root_path()
        self.zsl_kg = ZSLKG(task, root_path)

    def train(self, train_data_loader, val_data_loader, use_gpu):        
        # setup test graph (this will be used later)
        self.zsl_kg.train(train_data_loader, val_data_loader, use_gpu)

    def execute(self, unlabeled_data_loader, use_gpu):
        predictions = self.zsl_kg.execute(unlabeled_data_loader, use_gpu)

        return predictions

