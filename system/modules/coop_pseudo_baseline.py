from tqdm import tqdm
import logging
import numpy as np
import pandas as pd

import clip
import torch
from torch import nn
from PIL import Image
from accelerate import Accelerator
accelerator = Accelerator()

from ..utils import seed_worker, pseudolabel_top_k
from ..models import CustomTextEncoder, make_scheduler, TextPrefixModel
from ..modules import CoopBaseline

g = torch.Generator()
g.manual_seed(0)

log = logging.getLogger(__name__)


class CoopPseudoBaseline(CoopBaseline):
    def __init__(self, config, label_to_idx, 
                 classes, seen_classes, unseen_classes,
                 device, calibration_coefficient=None):
        """ This class define Coop baseline.

        :param config: dictionaries of prameters in models_config/coop_baseline_config.yml
        :param label_to_idx: dictionary (key, value):(class name, id)
        :param classes: list of class names
        :param seen_classes: list of seen classes' names
        :param unseen_classes: list of unseen classes' names
        :param device: device in use
        """
        super().__init__(config, label_to_idx, classes, 
                         seen_classes, unseen_classes,
                         device)      

    def create_training_dataset(self, train_data, unlabeled_data=None):
        """ This function create the dataset for training. Specifically, it
        merges pseudo-labels for unseen data and labeled data for seen classes.

        :param train_data: Dataset object - training seen classes (defined in zsl_jpl line 323)
        :param unlabeled_data: Dataset object - dataset of unlabeled data for 
                               unseen classes (defined in zsl_jpl line 328)
        """
        
        # Get pseudo-labels for unlabeled data from unseen classes
        train_unseen_dataset = pseudolabel_top_k(self.config.N_PSEUDOSHOTS,
                                                self.config.PROMPT_TEMPLATE,
                                                unlabeled_data,
                                                self.unseen_classes,
                                                self.transform,
                                                self.clip_model,
                                                self.label_to_idx,
                                                self.device)

        # Define the lists of traiing data from seen and unseen classes
        unseen_imgs = train_unseen_dataset.filepaths
        unseen_labs = train_unseen_dataset.labels

        seen_imgs = train_data.filepaths
        seen_labs = [self.label_to_idx[l] for l in train_data.labels]

        train_data.filepaths = list(unseen_imgs) + list(seen_imgs)
        train_data.labels = list(unseen_labs) + list(seen_labs)
        train_data.label_id = True    

        return train_data

