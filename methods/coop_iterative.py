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
from tqdm import tqdm

accelerator = Accelerator()

from data import CustomDataset
from models import CustomTextEncoder, TextPrefixModel
from methods import CoopPseudoBaseline
from utils import dataset_object, evaluate_predictions, make_scheduler, seed_worker

g = torch.Generator()
g.manual_seed(0)

log = logging.getLogger(__name__)

class IterativePseudoCoop(CoopPseudoBaseline):
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
            config, label_to_idx, classes, seen_classes, unseen_classes, device
        )

        self.data_folder = data_folder

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

        log.info(f"We select {self.config.N_PSEUDOSHOTS} per each unseen classes.")
        # Create a safe copy of labeled/unlabeled data
        original_train_data = copy.deepcopy(train_data)
        # log.info(f"Training data labels: {original_train_data.labels}")
        original_unlabeled_data = copy.deepcopy(unlabeled_data)
        # Original val
        original_val_data = copy.deepcopy(val_data)

        # Initialize here first batch of pseudo labels
        # Define training dataset
        log.info(f"BEFORE: {unlabeled_data.labels}")
        # log.info(f"BEFORE: {unlabeled_data.filepaths}")
        self.create_training_dataset(train_data, unlabeled_data)
        log.info(f"Labels unlabeled data: {unlabeled_data.labels}")

        for niter in range(1, num_iter + 1):
            log.info(f"Start {niter} round of training..")

            train_data.filepaths = [
                f for i, f in enumerate(original_train_data.filepaths)
            ]
            train_data.labels = [l for i, l in enumerate(original_train_data.labels)]
            self.update_training_set(train_data, unlabeled_data)

            # 1. Initialize teacher
            self.define_model(teacher=True)
            log.info(f"[TEACHER] Initialization..")

            # At this time the validation is composed only of seen classes. We can
            # try to expand it with pseudo-labels.
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

            # 3. Get teacher pseudo-labels
            log.info(f"[TEACHER] Collecting teacher pseudo-labels on unlabeled data..")
            # log.info(f"ORIGINAL UNLABELED DATA {original_unlabeled_data.filepaths}")
            pseudo_labels = self.get_pseudo_labels(
                original_unlabeled_data, teacher=True
            )

            # 4. Initialize student model
            log.info(f"[STUDENT] Initialization..")
            self.define_model(teacher=False)

            # 5. Train student
            log.info(f"[STUDENT] Start model training..")
            self.train_student(pseudo_labels)
            log.info(f"[STUDENT] Training completed.")

            # 6. Get new pseudo labels from student
            log.info(
                f"[STUDENT] Get student pseudo-labels for the next round of training."
            )
            if self.config.ALL_UNLABELED:
                n_per_class = int((niter + 1) * num_samples / n_unseen)
                if n_per_class * n_unseen <= len(original_unlabeled_data.filepaths):
                    # self.num_pseudo_labels_per_class =  n_per_class
                    self.config.N_PSEUDOSHOTS = n_per_class
                else:
                    # self.num_pseudo_labels_per_class =  math.floor(len(original_unlabeled_data.filepaths)/n_unseen)
                    self.config.N_PSEUDOSHOTS = math.floor(
                        len(original_unlabeled_data.filepaths) / n_unseen
                    )

            unlabeled_data = self.get_pseudo_labels(
                original_unlabeled_data, teacher=False
            )
            # Evaluate model at this point in time
            std_predictions = self.test_predictions(test_data, standard_zsl=True)

            # Submit predictions (standard)
            std_response = evaluate_predictions(
                std_predictions,
                test_labeled_files,
                test_labeles,
                self.unseen_classes,
                standard_zsl=True,
            )
            log.info(f"[ITERATION] ZSL accuracy: {std_response}")

            # Validate on test set (general)
            gen_predictions = self.test_predictions(test_data, standard_zsl=False)
            # Submit predictions (general)
            unseen_accuracy, seen_accuracy, harmonic_mean = evaluate_predictions(
                gen_predictions,
                test_labeled_files,
                test_labeles,
                self.unseen_classes,
                self.seen_classes,
                standard_zsl=False,
            )
            log.info(f"[ITERATION] Generalized ZSL results")
            log.info(f"[ITERATION] Accuracy seen classes: {seen_accuracy}")
            log.info(f"[ITERATION] Accuracy unseen classes: {unseen_accuracy}")
            log.info(f"[ITERATION] Harmonic mean: {harmonic_mean}")

        return t_best_val_accuracy, t_best_prompt
