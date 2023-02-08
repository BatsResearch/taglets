from tqdm import tqdm
import logging
import numpy as np
import pandas as pd

import clip
from ..data import CustomDataset
import torch
from sklearn.linear_model import LogisticRegression
from PIL import Image
from accelerate import Accelerator
from .vpt_baseline import VPTBaseline
from .vpt_pseudo_baseline import VPTPseudoBaseline
accelerator = Accelerator()

from ..utils import seed_worker
from ..models import CustomTextEncoder, make_scheduler

g = torch.Generator()
g.manual_seed(0)

log = logging.getLogger(__name__)


class TwoStageClassifier(object):
    def __init__(self, config, label_to_idx, 
                 classes, seen_classes, unseen_classes,
                 device, calibration_coefficient=None):

        self.config = config
        self.classes = classes
        self.seen_classes = seen_classes
        self.unseen_classes = unseen_classes
        self.label_to_idx = label_to_idx
        self.calibration_coefficient = calibration_coefficient

        # self.linear_layer = LinearClassifier(
        #     output_dim=2 if config.FIRST_STAGE_TYPE == 'binary' else len(self.seen_classes) + 1)

        seen_to_idx = {c:idx for idx, c in enumerate(self.seen_classes)}
        self.idx_to_real = {seen_to_idx[c]:self.label_to_idx[c] for c in self.seen_classes}
        self.real_to_idx = {self.label_to_idx[c]:seen_to_idx[c] for c in self.seen_classes}


        self.device = device
        self.clip_model, self.transform = clip.load(self.config.VIS_ENCODER, 
                                                    device=self.device)
        self.template = self.config.PROMPT_TEMPLATE

        if torch.cuda.is_available():
            self.text_encoder = CustomTextEncoder(self.clip_model, self.device, 
                                                  torch.float16).to(self.device)
        else:
            self.text_encoder = CustomTextEncoder(self.clip_model, self.device,
                                                  torch.half).to(self.device)

    def reprocess_data(self, train_data_seen, train_data_unseen, val_data_seen, unseen_train_split=.8):
        train_data_seen.transform = self.transform
        train_data_unseen.transform = self.transform
        val_data_seen.transform = self.transform
        # Split the train_data_unseen into two parts, so we have some val data
        train_data_unseen, val_data_unseen = torch.utils.data.random_split(train_data_unseen, [int(len(
            train_data_unseen) * unseen_train_split), len(train_data_unseen) - int(len(train_data_unseen) * unseen_train_split)])
        train_data_unseen = train_data_unseen.dataset
        val_data_unseen = val_data_unseen.dataset

        if self.config.FIRST_STAGE_TYPE == "binary":
            seen_labels_train = ["seen_class" for _ in train_data_seen.labels]
            seen_labels_val = ["seen_class" for _ in val_data_seen.labels]
            label_map = {"seen_class": 0, "unknown_class": 1}
        else:
            seen_labels_train = train_data_seen.labels
            seen_labels_val = val_data_seen.labels
            label_map = {**train_data_seen.label_map, **{"unknown_class": len(set(seen_labels_train))}}

        unseen_labels_train = ["unknown_class" for _ in train_data_unseen]
        unseen_labels_val = ["unknown_class" for _ in val_data_unseen]
        
        # Create new datasets
        train_labels = [str(s) for s in seen_labels_train] + [str(s) for s in unseen_labels_train]
        val_labels = [str(s) for s in seen_labels_val] + [str(s) for s in unseen_labels_val]
        combined_train_dataset = CustomDataset(train_data_seen.filepaths + train_data_unseen.filepaths,
                                         "", self.transform, augmentations=None,
                                         train=True, labels=train_labels, label_id=False,
                                         label_map=label_map)
        combined_train_dataset.filepaths = train_data_seen.filepaths + train_data_unseen.filepaths
        combined_val_dataset = CustomDataset(val_data_seen.filepaths + val_data_unseen.filepaths,
                                         "", self.transform, augmentations=None,
                                         train=True, labels=val_labels, label_id=False,
                                         label_map=label_map)
        combined_val_dataset.filepaths = val_data_seen.filepaths + val_data_unseen.filepaths

        return combined_train_dataset, combined_val_dataset

    def train_vpt(self, train_data, val_data):
        self.config.EPOCHS = self.config.STAGE_TWO_VPT_EPOCHS
        # observe we set unseen_classes = self.seen_classes, since we only want
        # to use seen classes at test time
        vpt = VPTBaseline(self.config, self.label_to_idx, self.classes,
                          self.seen_classes, self.seen_classes, self.device)
        val_accuracy, optimal_prompt = vpt.train(train_data, val_data)
        log.info("Supervised VPT val accuracy: {}".format(val_accuracy))
        return vpt, optimal_prompt

    # train data and val data are both from unseen c
    def train_pseudo_vpt(self, train_data):
        # observe we set classes = self.seen_classes, since we only want to
        # train/test on seen classes
        self.config.EPOCHS = self.config.STAGE_TWO_PSEUDO_EPOCHS
        dummy_dset = CustomDataset([], "", self.transform, augmentations=None,
                                         train=True, labels=[], label_id=False,
                                         label_map={})
        pseudo_vpt = VPTPseudoBaseline(self.config, self.label_to_idx, self.classes,
                                       self.seen_classes, self.unseen_classes, self.device)
        val_accuracy, optimal_prompt = pseudo_vpt.train(dummy_dset, train_data, dummy_dset, None, only_unlabelled=True)
        log.info("Unsupervised Pseudo-VPT val accuracy: {}".format(val_accuracy))
        return pseudo_vpt, optimal_prompt

    def train(self, train_data_seen, train_data_unseen, val_data_seen, unseen_train_split=.8):
        # create new datasets with "unknown_class" labels
        # train_dataset and val_dataset contain both seen and unseen classes
        train_dataset, val_dataset = self.reprocess_data(
            train_data_seen, train_data_unseen, val_data_seen, unseen_train_split)
        # train the first stage linear probe (TODO: is prompt better?)
        self.train_first_stage(train_dataset, val_dataset)
        # train supervised vpt
        self.vpt, self.vpt_prompt = self.train_vpt(train_data_seen, val_data_seen)
        # train unsuperivsed pseudo-vpt
        self.pseudo_vpt, self.pseudo_vpt_prompt = self.train_pseudo_vpt(train_data_unseen)


    # train a linear classifier on the clip embeddings
    # discriminate between seen and unseen classes
    def train_first_stage(self, train_data, val_data):
        train_data.transform = self.transform
        val_data.transform = self.transform
        train_loader = torch.utils.data.DataLoader(train_data,
                                                   batch_size=self.config.BATCH_SIZE,
                                                   shuffle=True, worker_init_fn=seed_worker,
                                                   generator=g)
        val_loader = torch.utils.data.DataLoader(val_data,
                                                 batch_size=self.config.BATCH_SIZE)
        train_features = []
        train_labels = []

        for i, (img, _, _, label, img_path) in enumerate(tqdm(train_loader)):
            with torch.no_grad():
                clip_embeddings = self.clip_model.encode_image(img.to(self.device))
            train_features.append(clip_embeddings.detach().cpu())
            train_labels.append(label.cpu())

        train_features = torch.cat(train_features).cpu().numpy()
        train_labels = torch.cat(train_labels).cpu().numpy()

        # Code obtained from https://github.com/openai/CLIP
        # Perform logistic regression
        # TODO: Perform hyperparam sweep to obtain best C param
        classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000)
        classifier.fit(train_features, train_labels)

        val_features = []
        val_labels = []

        for i, (img, _, _, label, img_path) in enumerate(tqdm(val_loader)):
            with torch.no_grad():
                clip_embeddings = self.clip_model.encode_image(img.to(self.device))
            val_features.append(clip_embeddings.detach())
            val_labels.append(label)

        val_features = torch.cat(val_features).cpu().numpy()
        val_labels = torch.cat(val_labels).cpu().numpy()
        predictions_t = classifier.predict(train_features)
        # Evaluate using the logistic regression classifier
        predictions = classifier.predict(val_features)
        accuracy = np.mean((val_labels == predictions).astype(float)) * 100.
        correct_binary = 0
        self.unknown_class_idx = train_data.label_map["unknown_class"]
        for idx in range(len(val_labels)):
            if val_labels[idx] == train_data.label_map["unknown_class"]:
                if predictions[idx] == train_data.label_map["unknown_class"]:
                    correct_binary += 1
            else:
                if predictions[idx] != train_data.label_map["unknown_class"]:
                    correct_binary += 1

        accuracy_binary = correct_binary / len(val_labels) * 100.
        log.info(f"Val accuracy = {accuracy:.3f}")
        log.info(f"Val accuracy BINARY = {accuracy_binary:.3f}")
        self.first_stage = classifier
            

    def test_predictions(self, test_set, standard_zsl=False):
        # get the end of the file path
        if standard_zsl:
            # Only using unseen classes - regular Pseudo-VPT
            return self.pseudo_vpt.test_predictions(test_set, standard_zsl=True)
        else:
            # Use two stage approach
            # Predict if examples are in seen or unseen classes
            test_set.transform = self.transform
            test_loader = torch.utils.data.DataLoader(test_set, 
                                                    batch_size=self.config.BATCH_SIZE)
            test_features = []
            with torch.no_grad():
                for images, _, _, img_path in tqdm(test_loader):
                    features = self.clip_model.encode_image(images.to(self.device))
                    test_features.append(features)
            test_features = torch.cat(test_features).cpu().numpy()
            set_predictions = self.first_stage.predict(test_features)
            unseen_classes_fps = []
            seen_classes_fps = []
            for idx, pred in enumerate(set_predictions):
                if pred == self.unknown_class_idx:
                    unseen_classes_fps.append(test_set.filepaths[idx])
                else:
                    seen_classes_fps.append(test_set.filepaths[idx])
            unseen_classes_dset = CustomDataset(unseen_classes_fps, "", self.transform, augmentations=None,
                                                train=False, labels=None, label_id=False, label_map=test_set.label_map)
            seen_classes_dset = CustomDataset(seen_classes_fps, "", self.transform, augmentations=None,
                                              train=False, labels=None, label_id=False, label_map=test_set.label_map)
            unseen_classes_dset.filepaths = unseen_classes_fps
            seen_classes_dset.filepaths = seen_classes_fps
            # Use VPT to predict seen classes, psedu-VPT to predict unseen classes
            seen_classes_preds = self.vpt.test_predictions(seen_classes_dset, standard_zsl=True)
            unseen_classes_preds = self.pseudo_vpt.test_predictions(unseen_classes_dset, standard_zsl=True)
            return pd.concat([seen_classes_preds, unseen_classes_preds], ignore_index=True)
