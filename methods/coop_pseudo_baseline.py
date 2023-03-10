import logging

import clip
import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from PIL import Image
from torch import nn
from tqdm import tqdm

accelerator = Accelerator()

from models import CustomTextEncoder, TextPrefixModel
from methods import CoopBaseline
from utils import make_scheduler, pseudolabel_top_k, seed_worker


g = torch.Generator()
g.manual_seed(0)

log = logging.getLogger(__name__)


class CoopPseudoBaseline(CoopBaseline):
    def __init__(
        self,
        config,
        label_to_idx,
        classes,
        seen_classes,
        unseen_classes,
        device,
        calibration_coefficient=None,
    ):
        """This class define Coop baseline.

        :param config: dictionaries of prameters in models_config/coop_baseline_config.yml
        :param label_to_idx: dictionary (key, value):(class name, id)
        :param classes: list of class names
        :param seen_classes: list of seen classes' names
        :param unseen_classes: list of unseen classes' names
        :param device: device in use
        """
        super().__init__(
            config, label_to_idx, classes, seen_classes, unseen_classes, device
        )

    def create_training_dataset(self, train_data, unlabeled_data=None):
        """This function create the dataset for training. Specifically, it
        merges pseudo-labels for unseen data and labeled data for seen classes.

        :param train_data: Dataset object - training seen classes (defined in zsl_jpl line 323)
        :param unlabeled_data: Dataset object - dataset of unlabeled data for
                               unseen classes (defined in zsl_jpl line 328)
        """

        # Get pseudo-labels for unlabeled data from unseen classes
        train_unseen_dataset = pseudolabel_top_k(
            self.config.DATASET_NAME,
            self.config.N_PSEUDOSHOTS,
            self.config.PROMPT_TEMPLATE,
            unlabeled_data,
            self.unseen_classes,
            self.transform,
            self.clip_model,
            self.label_to_idx,
            self.device,
            self.config.VIS_ENCODER,
        )

        # Define the lists of traiing data from seen and unseen classes
        unseen_imgs = train_unseen_dataset.filepaths
        unseen_labs = train_unseen_dataset.labels

        # Use a portion of the pseudo-labeled data to build a validation set
        if self.config.N_PSEUDOSHOTS >= 10:
            np.random.seed(self.config.validation_seed)
            train_indices = np.random.choice(
                range(len(unseen_imgs)),
                size=int(len(unseen_imgs) * self.config.ratio_train_val),
                replace=False,
            )
            val_indices = list(
                set(range(len(unseen_imgs))).difference(set(train_indices))
            )

            self.val_unseen_files = np.array(unseen_imgs)[val_indices]
            self.val_unseen_labs = np.array(unseen_labs)[val_indices]

            unseen_imgs = list(np.array(unseen_imgs)[train_indices])
            unseen_labs = list(np.array(unseen_labs)[train_indices])

        else:
            self.val_unseen_files = None
            self.val_unseen_labs = None

        seen_imgs = train_data.filepaths
        seen_labs = [self.label_to_idx[l] for l in train_data.labels]

        self.balance_param = len(seen_imgs) / len(unseen_imgs)

        train_data.filepaths = list(unseen_imgs) + list(seen_imgs)
        train_data.labels = list(unseen_labs) + list(seen_labs)
        train_data.label_id = True

        return train_data

    def define_loss_function(self, logits, labs):
        # loss_ce_seen = self.loss_func(logits, labs)
        # loss_unseen = self.loss_disambiguate(logits, labs)

        loss_ce_seen = self.cross_entropy(logits, labs, self.seen_classes)
        # log.info(f"CE seen classes: {loss_ce_seen}")

        loss_ce_unseen = self.cross_entropy(logits, labs, self.unseen_classes)
        # log.info(f"CE unseen classes: {loss_ce_unseen}")

        return loss_ce_seen + self.balance_param * loss_ce_unseen

    def cross_entropy(self, logits, labels, classes):
        """This loss computes the probability mass on the
        opposite set of classes for each sample.

        :param logits: continuous vector
        :param labels: class ids
        """

        ids = [self.label_to_idx[c] for c in classes]

        # Get indices of unseen and seen samples in the batch
        samples = []

        for idx, l in enumerate(labels):
            if l in ids:
                samples.append(idx)

        # Get logit sums on unseen samples
        if samples:
            error = self.loss_func(logits[samples], labels[samples])
        else:
            error = 0

        return error

    def reindex_predicted_labels(self, idx_preds, only_unlabelled=False):
        """This function returns the correct index of predictions to compute
        model's accuracy.

        :param idx_pred: list of predictions ids
        :param only_unlabelled: boolean. It is True if the training only involves
                                pseudo-labeled unseen data
        """

        if only_unlabelled:
            return [self.unseen_classes[i.item()] for i in idx_preds]
        else:
            return [self.classes[i.item()] for i in idx_preds]

    def reindex_true_labels(self, label, only_unlabelled=False):
        """This function returns the correct index of true labels.

        :param label: list of labels from data loader
        :param only_unlabelled: boolean. It is True if the training only involves
                                pseudo-labeled unseen data
        """

        if only_unlabelled:
            return torch.tensor(
                [self.unseen_classes.index(self.classes[l.item()]) for l in label]
            )
        else:
            return torch.tensor([l for l in label])
