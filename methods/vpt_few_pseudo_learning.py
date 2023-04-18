import logging

import clip
import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from PIL import Image
from torch import nn

accelerator = Accelerator()

from models import CustomImageEncoder
from methods import VPT
from utils import make_scheduler, pseudolabel_top_k


log = logging.getLogger(__name__)


class VptFPL(VPT):
    def __init__(
        self, 
        config, 
        label_to_idx, 
        classes, 
        seen_classes, 
        unseen_classes, 
        device
    ):
        """This class defines few-pseudo-learning (FPL) VPT's training and evaluation.

        :param config: dictionaries of prameters in models_config/vpt_baseline_config.yml
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
        """This function creates the dataset for training. Specifically, it
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
            self.config.SPLIT_SEED,
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

    def fixed_iterative_train(
        self,
        train_data,
        val_data,
        unlabeled_data,
    ):
        # Number of total iterations to cover all unlabeled data
        num_iter = int(100/self.config.STEP_QUANTILE)
        num_samples = int(len(unlabeled_data) / num_iter)
        # Initialize the number of pseudo-labels per class
        n_per_class = int(num_samples / len(self.classes))
        n_unseen = len(self.unseen_classes)
        # if n_per_class * n_unseen <= len(unlabeled_data.filepaths):
        #     # self.num_pseudo_labels_per_class =  n_per_class
        #     self.config.N_PSEUDOSHOTS = n_per_class
        # else:
        #     # self.num_pseudo_labels_per_class =  math.floor(len(unlabeled_data.filepaths)/n_unseen)
        #     self.config.N_PSEUDOSHOTS = math.floor(
        #         len(unlabeled_data.filepaths) / n_unseen
        #     )

        log.info(f"We select {self.config.N_PSEUDOSHOTS} pseudolabel per each unseen classes.")
        log.info(f"The number of unseen classes is: {len(self.unseen_classes)}.")
        log.info(f"Thus we expect an initial number of pseudo labeles equal to {len(self.unseen_classes) * self.config.N_PSEUDOSHOTS}.")

        # Create a safe copy of labeled/unlabeled data
        original_train_data = copy.deepcopy(train_data)
        # log.info(f"Training data labels: {original_train_data.labels}")
        original_unlabeled_data = copy.deepcopy(unlabeled_data)
        # Original val
        original_val_data = copy.deepcopy(val_data)

        # 1. Get training data  (also get pseudo-labeles from CLIP)
        self.create_training_dataset(train_data, unlabeled_data)
        log.info(f"The original train data has size: {len(original_train_data.filepaths)}.")
        log.info(f"Only with unlabeled is: {len(unlabeled_data.filepaths)}.")
        log.info(f"Current train data is: {len(train_data.filepaths)}.")

        # Save pseudolabels
        log.info(f"Saving pseudo-labels for init")
        save_pseudo_labels(
            unlabeled_data.filepaths, 
            unlabeled_data.labels, 
            self.config, 
            0,
            teacher=False,
        )
        log.info(f"Unlabeled is: {len(unlabeled_data.filepaths)}.")
        
        for niter in range(1, num_iter + 1):
            log.info(f"Start {niter} round of training..")

            train_data.filepaths = [
                f for i, f in enumerate(original_train_data.filepaths)
            ]
            train_data.labels = [l for i, l in enumerate(original_train_data.labels)]
           
            log.info(f"Unlabeled is {len(unlabeled_data.filepaths)} in iter: {niter}.")
            self.update_training_set(train_data, unlabeled_data)

            # 2. Define model
            self.define_model()
            log.info(f"[MODEL] Initialization iter {niter}")

            # 3. Train model
            log.info(f"[MODEL] Start model training iter {niter}..")
            t_best_val_accuracy, t_best_prompt = self.train(
                train_data, val_data
            )
            log.info(f"[MODEL] Training completed iter {niter}.")

            log.info(f"[MODEL] Collecting model pseudo-labels on unlabeled data..")
            unlabeled_data = self.get_pseudo_labels(
                original_unlabeled_data
            )

             # Save pseudolabels
            log.info(f"Saving pseudo-labels for iteration {niter}")
            save_pseudo_labels(
                unlabeled_data.filepaths, 
                unlabeled_data.labels, 
                self.config, 
                niter,
                teacher=True,
            )

            save_parameters(
                t_best_prompt,
                self.config, 
                teacher=True, 
                iteration=niter
            )

        return t_best_val_accuracy, t_best_prompt

    def iterative_train(
        self,
        train_data,
        val_data,
        unlabeled_data,
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

        # 1. Get training data  (also get pseudo-labeles from CLIP)
        unlabeled_data = self.create_training_dataset(train_data, unlabeled_data)
        log.info(f"The original train data has size: {len(original_train_data.filepaths)}.")
        log.info(f"Only with unlabeled is: {len(unlabeled_data.filepaths)}.")
        log.info(f"Current train data is: {len(train_data.filepaths)}.")

        # Save pseudolabels
        log.info(f"Saving pseudo-labels for init")
        save_pseudo_labels(
            unlabeled_data.filepaths, 
            unlabeled_data.labels, 
            self.config, 
            0,
            teacher=False,
        )
        log.info(f"Unlabeled is: {len(unlabeled_data.filepaths)}.")
        
        for niter in range(1, num_iter + 1):
            log.info(f"Start {niter} round of training..")

            train_data.filepaths = [
                f for i, f in enumerate(original_train_data.filepaths)
            ]
            train_data.labels = [l for i, l in enumerate(original_train_data.labels)]


            log.info(f"Unlabeled is {len(unlabeled_data.filepaths)} in iter: {niter}.")
            self.update_training_set(train_data, unlabeled_data)

            # 2. Define model
            self.define_model()
            log.info(f"[MODEL] Initialization iter {niter}")

            # Validation with seen and unseen.
            if self.val_unseen_files is not None:
                seen_imgs = original_val_data.filepaths
                seen_labs = [self.label_to_idx[l] for l in original_val_data.labels]

                unseen_imgs = list(self.val_unseen_files)
                unseen_labs = list(self.val_unseen_labs)

                val_data.filepaths = list(unseen_imgs) + list(seen_imgs)
                val_data.labels = list(unseen_labs) + list(seen_labs)
                val_data.label_id = True

            # 3. Train model
            log.info(f"[MODEL] Start model training iter {niter}..")
            t_best_val_accuracy, t_best_prompt = self.train(
                train_data, val_data
            )
            log.info(f"[MODEL] Training completed iter {niter}.")

            # 4. Generate new pseudo-labels
            # Define number of pseudo-labels to collect
            n_per_class = int((niter + 1) * num_samples / n_unseen)
            if n_per_class * n_unseen <= len(original_unlabeled_data.filepaths):
                self.config.N_PSEUDOSHOTS = n_per_class
            else:
                # We are making a stong assumption about the distribution of unlabeled data
                self.config.N_PSEUDOSHOTS = math.floor(
                    len(original_unlabeled_data.filepaths) / n_unseen
                )

            log.info(f"We select {self.config.N_PSEUDOSHOTS} pseudolabel per each unseen classes.")
            log.info(f"The number of unseen classes is: {len(self.unseen_classes)}.")
            log.info(f"Thus, for iter {niter} we expect a number of pseudo labeles equal to {len(self.unseen_classes) * self.config.N_PSEUDOSHOTS}.")

            log.info(f"[MODEL] Collecting model pseudo-labels on unlabeled data..")
            unlabeled_data = self.get_pseudo_labels(
                original_unlabeled_data
            )

             # Save pseudolabels
            log.info(f"Saving pseudo-labels for iteration {niter}")
            save_pseudo_labels(
                unlabeled_data.filepaths, 
                unlabeled_data.labels, 
                self.config, 
                niter,
                teacher=True,
            )

            save_parameters(
                t_best_prompt,
                self.config, 
                teacher=True, 
                iteration=niter
            )

            # self.config.EPOCHS += 10

        return t_best_val_accuracy, t_best_prompt

    def define_loss_function(self, logits, labs, teacher=False):

        loss_ce_seen = self.cross_entropy(logits, labs, self.seen_classes)
        loss_ce_unseen = self.cross_entropy(logits, labs, self.unseen_classes)

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

        if samples:
            error = self.loss_func(logits[samples], labels[samples])
        else:
            error = 0

        return error

    def define_textual_prompts(self, only_unlabelled=None, validation=False):
        """This function returns the textual prompts. You can modify the list
        of classes of interest.

        :param only_unlabelled: boolean. It is True if the training only involves
                                pseudo-labeled unseen data
        """

        if only_unlabelled:
            return [
                self.template.format(" ".join(i.split("_")))
                for i in self.unseen_classes
            ]
        else:
            if validation:
                return [
                    self.template.format(" ".join(i.split("_")))
                    for i in self.seen_classes
                ]
            else:
                return [
                    self.template.format(" ".join(i.split("_"))) for i in self.classes
                ]

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

    def update_training_set(self, train_data, unlabeled_data):
        
        # Get pseudo-labels for unlabeled data from unseen classes
        train_unseen_dataset = unlabeled_data
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

        # Specify train labeled data
        seen_imgs = train_data.filepaths
        seen_labs = [self.label_to_idx[l] for l in train_data.labels]

        # Weigth the classes representativeness
        self.balance_param = len(seen_imgs) / len(unseen_imgs)

        train_data.filepaths = list(unseen_imgs) + list(seen_imgs)
        train_data.labels = list(unseen_labs) + list(seen_labs)
        train_data.label_id = True
        log.info(f"UPDATE DATASET: size = {len(train_data.filepaths)}")
        log.info(f"UPDATE UNSEEN DATASET: size = {len(unseen_imgs)}")
        log.info(f"UPDATE SEEN DATASET: size = {len(seen_imgs)}")


    def get_pseudo_labels(self, unlabeled_examples, teacher=True):
        log.info(f"Num unlabeled data: {len(unlabeled_examples)}")
        # Get prediction of teacher on unlabeled data
        std_preds = self.test_predictions(
            unlabeled_examples, standard_zsl=True, teacher=teacher
        )

        DatasetObject = dataset_object(self.config.DATASET_NAME)
        # 4. Take top-16 pseudo-labels to finetune the student
        pseudo_unseen_examples = DatasetObject(
            std_preds["id"],
            self.data_folder,
            transform=self.transform,
            augmentations=None,
            train=True,
            labels=None,
            label_map=self.label_to_idx,
            class_folder=True,
            original_filepaths=unlabeled_examples.filepaths,
        )

        pseudo_labels = self.assign_pseudo_labels(
            self.config.N_PSEUDOSHOTS, pseudo_unseen_examples
        )

        return pseudo_labels

    def assign_pseudo_labels(self, k, unlabeled_data, teacher=True):
        # Define text queries
        # prompts = [f"{self.template}{' '.join(i.split('_'))}" \
        #             for i in self.unseen_classes]
        prompts = [
            self.template.format(" ".join(i.split("_"))) for i in self.unseen_classes
        ]
        log.info(f"Number of prompts: {len(prompts)}")

        # Encode text
        text = clip.tokenize(prompts).to(self.device)
        text_features = self.clip_model.encode_text(text)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # to find the top k for each class, each class has it's own "leaderboard"
        top_k_leaderboard = {
            self.label_to_idx[self.unseen_classes[i]]: []
            for i in range(len(self.unseen_classes))
        }  # maps class idx -> (confidence, image_path) tuple

        for img_path in unlabeled_data.filepaths:
            # log.info(f"IMAGEPATH: {img_path}")
            img = Image.open(img_path).convert("RGB")
            img = torch.unsqueeze(self.transform(img), 0).to(self.device)
            with torch.no_grad():
                if teacher:
                    image_features = self.teacher(img)
                else:
                    image_features = self.student(img)
                image_features = image_features / image_features.norm(
                    dim=-1, keepdim=True
                )
                # cosine similarity as logits

            logit_scale = self.clip_model.logit_scale.exp()
            logits = logit_scale * image_features @ text_features.t()
            probs = logits.softmax(dim=-1)
            idx_preds = torch.argmax(logits, dim=1)
            pred_id = idx_preds.item()
            pred = self.label_to_idx[self.unseen_classes[idx_preds.item()]]

            """if predicted class has empty leaderboard, or if the confidence is high
            enough for predicted class leaderboard, add the new example
            """
            prob_score = probs[0][pred_id]
            if len(top_k_leaderboard[pred]) < k:
                top_k_leaderboard[pred].append((prob_score, img_path))
            elif (
                top_k_leaderboard[pred][-1][0] < prob_score
            ):  # if the confidence in predicted class "qualifies" for top-k
                # default sorting of tuples is by first element
                top_k_leaderboard[pred] = sorted(
                    top_k_leaderboard[pred] + [(probs[0][pred_id], img_path)],
                    reverse=True,
                )[:k]
            else:
                # sort the other classes by confidence score
                order_of_classes = sorted(
                    [
                        (probs[0][j], j)
                        for j in range(len(self.unseen_classes))
                        if j != pred_id
                    ],
                    reverse=True,
                )
                for score, index in order_of_classes:
                    index_dict = self.label_to_idx[self.unseen_classes[index]]
                    # log.info(f"{classnames[index]}")
                    # log.info(f"{index_dict}")
                    if len(top_k_leaderboard[index_dict]) < k:
                        top_k_leaderboard[index_dict].append(
                            (probs[0][index], img_path)
                        )
                    elif top_k_leaderboard[index_dict][-1][0] < probs[0][index]:
                        # default sorting of tuples is by first element
                        top_k_leaderboard[index_dict] = sorted(
                            top_k_leaderboard[index_dict]
                            + [((probs[0][index], img_path))],
                            reverse=True,
                        )[:k]

        new_imgs = []
        new_labels = []
        # loop through, and rebuild the dataset
        for index, leaderboard in top_k_leaderboard.items():
            new_imgs += [tup[1] for tup in leaderboard]
            new_labels += [index for _ in leaderboard]

        unlabeled_data.filepaths = new_imgs
        unlabeled_data.labels = new_labels
        unlabeled_data.label_id = True

        return unlabeled_data
