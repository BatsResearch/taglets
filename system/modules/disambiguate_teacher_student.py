import copy
import math
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

from ..data import CustomDataset
from ..utils import seed_worker, pseudolabel_top_k
from ..models import CustomImageEncoder, make_scheduler, ImagePrefixModel
from ..modules import TeacherStudent

g = torch.Generator()
g.manual_seed(0)

log = logging.getLogger(__name__)

class DisambiguateTeacherStudent(TeacherStudent):
    def __init__(self, config, label_to_idx, data_folder, 
                 classes, seen_classes, unseen_classes,
                 device):
        
        super().__init__(config, label_to_idx, data_folder,
                         classes, seen_classes, unseen_classes,
                         device) 

        self.loss_disambiguate = self.disambiguation_loss
        log.info(f"Verify subclass work: {self.loss_disambiguate}")

    def disambiguation_loss(self, logits, labels):
        """ This loss computes the probability mass on the
        opposite set of classes for each sample.
        
        :param logits: continuous vector
        :param labels: class ids
        """
        softmax = nn.Softmax(dim=1)
        logits = softmax(logits)
        #log.info(f"LOGITS: {logits}")
        seen_ids = [self.label_to_idx[c] for c in self.seen_classes]
        unseen_ids = [self.label_to_idx[c] for c in self.unseen_classes]

        # Get indices of unseen and seen samples in the batch
        unseen_samples = [] 
        seen_samples = []
        for idx, l in enumerate(labels):
            if l in unseen_ids:
                unseen_samples.append(idx)
            elif l in seen_ids:
                seen_samples.append(idx)
        # Get logit sums on unseen samples
        if unseen_samples:
            error_unseen = torch.sum(logits[unseen_samples][:, seen_ids])
        else:
            error_unseen = 0
        if seen_samples:
            error_seen = torch.sum(logits[seen_samples][:,unseen_ids])
        else:
            error_seen = 0
        return error_unseen + error_seen


    def define_loss_function(self, logits, labs, teacher=False):
        
        if teacher:
            loss_ce = self.teacher_loss_func(logits, labs)
            loss_dis = self.loss_disambiguate(logits, labs)
            return loss_ce + loss_dis
        else:
            return self.student_loss_func(logits, labs)
        