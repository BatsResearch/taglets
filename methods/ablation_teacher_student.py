import copy
import logging
import math

import clip
import numpy as np
import pandas as pd
import scipy.stats as st
import torch
from accelerate import Accelerator
from PIL import Image
from torch import nn


accelerator = Accelerator()

from data import CustomDataset
from models import CustomImageEncoder, ImagePrefixModel
from methods import TeacherStudent
from utils import (
    dataset_object, 
    evaluate_predictions, 
    make_scheduler, 
    seed_worker, 
    save_parameters,
    save_pseudo_labels,
)

g = torch.Generator()
g.manual_seed(0)

log = logging.getLogger(__name__)


class AblationTeacherStudent(TeacherStudent):
    def __init__(
        self,
        config,
        label_to_idx,
        data_folder,
        classes,
        seen_classes,
        unseen_classes,
        device,
    ):
        super().__init__(
            config, label_to_idx, data_folder, classes, seen_classes, unseen_classes, device
        )


    def train(
        self,
        train_data,
        val_data,
        unlabeled_data,
        test_data,
        test_labeled_files,
        test_labeles,
    ):
        # Number of total iterations to cover all unlabeled data
        num_iter = int(100/self.config.STEP_QUANTILE)
        num_samples = int(len(unlabeled_data) / num_iter)
        # Initialize the number of pseudo-labels per class
        n_per_class = int(num_samples / len(self.unseen_classes))
        n_unseen = len(self.unseen_classes)
        if n_per_class * n_unseen <= len(unlabeled_data.filepaths):
            # self.num_pseudo_labels_per_class =  n_per_class
            self.config.N_PSEUDOSHOTS = n_per_class
        else:
            # self.num_pseudo_labels_per_class =  math.floor(len(unlabeled_data.filepaths)/n_unseen)
            self.config.N_PSEUDOSHOTS = math.floor(
                len(unlabeled_data.filepaths) / n_unseen
            )

        log.info(f"We select {self.config.N_PSEUDOSHOTS} pseudolabel per each unseen classes.")
        log.info(f"The number of unseen classes is: {len(self.unseen_classes)}.")
        log.info(f"Thus we expect an initial number of pseudo labeles equal to {len(self.unseen_classes) * self.config.N_PSEUDOSHOTS}.")
        # Create a safe copy of labeled/unlabeled data
        original_train_data = copy.deepcopy(train_data)
        # log.info(f"Training data labels: {original_train_data.labels}")
        original_unlabeled_data = copy.deepcopy(unlabeled_data)
        # Original val
        original_val_data = copy.deepcopy(val_data)

        # Initialize here first batch of pseudo labels
        self.create_training_dataset(train_data, unlabeled_data)
        log.info(f"The original train data has size: {len(original_train_data.filepaths)}.")
        log.info(f"Plus: {len(unlabeled_data.filepaths)}.")

        for niter in range(1, num_iter + 1):
            log.info(f"Start {niter} round of training..")

            train_data.filepaths = [
                f for i, f in enumerate(original_train_data.filepaths)
            ]
            train_data.labels = [l for i, l in enumerate(original_train_data.labels)]

            # Save pseudolabels
            log.info(f"Saving pseudo-labels for iteration {niter}")
            save_pseudo_labels(
                unlabeled_data.filepaths, 
                unlabeled_data.labels, 
                self.config, 
                niter,
                teacher=False,
            )
            
            self.update_training_set(train_data, unlabeled_data)

            # 1. Initialize teacher
            self.define_model(teacher=True)
            log.info(f"[TEACHER] Initialization..")

            # Validation with seen and unseen.
            if self.val_unseen_files is not None:
                seen_imgs = original_val_data.filepaths
                seen_labs = [self.label_to_idx[l] for l in original_val_data.labels]

                unseen_imgs = list(self.val_unseen_files)
                unseen_labs = list(self.val_unseen_labs)

                val_data.filepaths = list(unseen_imgs) + list(seen_imgs)
                val_data.labels = list(unseen_labs) + list(seen_labs)
                val_data.label_id = True

            # 2. Train teacher with labeled seen and pseudo-labeled unseen
            log.info(f"[TEACHER] Start model training..")
            t_best_val_accuracy, t_best_prompt = self.train_teacher(
                train_data, val_data
            )
            log.info(f"[TEACHER] Training completed.")


            # Exploit all the available unlabeled data
            if self.config.ALL_UNLABELED:
                n_per_class = int((niter + 1) * num_samples / n_unseen)
                if n_per_class * n_unseen <= len(original_unlabeled_data.filepaths):
                    self.config.N_PSEUDOSHOTS = n_per_class
                else:
                    # We are making a stong assumption about the distribution of unlabeled data
                    self.config.N_PSEUDOSHOTS = math.floor(
                        len(original_unlabeled_data.filepaths) / n_unseen
                    )
            
            # 3. Get teacher pseudo-labels
            log.info(f"[TEACHER] Collecting teacher pseudo-labels on unlabeled data..")
            unlabeled_data = self.get_pseudo_labels(
                original_unlabeled_data, teacher=True
            )

            save_pseudo_labels(
                unlabeled_data.filepaths, 
                unlabeled_data.labels, 
                self.config, 
                niter,
                teacher=True,
            )

            save_parameters(t_best_prompt, self.config, teacher=True, iteration=niter)
            save_parameters(s_prompt, self.config, teacher=False, iteration=niter)

        return t_best_val_accuracy, t_best_prompt