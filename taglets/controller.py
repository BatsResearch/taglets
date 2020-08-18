from .data import SoftLabelDataset
from .modules import FineTuneModule, PrototypeModule, TransferModule, MultiTaskModule, ZSLKGModule
from .pipeline import EndModel, TagletExecutor

import logging
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset

####################################################################
# We configure logging in the main class of the application so that
# subprocesses inherit the same configuration. This would have to be
# redesigned if used as part of a larger application with its own
# logging configuration
####################################################################
logger = logging.getLogger()
logger.level = logging.INFO
stream_handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
####################################################################
# End of logging configuration
####################################################################

log = logging.getLogger(__name__)


class Controller:
    """
    Manages training and execution of taglets, as well as training EndModels
    """
    def __init__(self, task):
        self.task = task
        self.end_model = None

    def train_end_model(self):
        """
        Executes a training pipeline end-to-end, turning a Task into an EndModel
        :return: A trained EndModel
        """
        # Gets datasets
        labeled = self.task.get_labeled_train_data()
        unlabeled = self.task.get_unlabeled_data(False)
        val = self.task.get_validation_data()

        unlabeled_images_labels = []
        if unlabeled is not None:
            # Initializes taglet-creating modules
            modules = self._get_taglets_modules()

            for module in modules:
                log.info("Training %s module", module.__class__.__name__)
                module.train_taglets(labeled, val)
                log.info("Finished training %s module", module.__class__.__name__)
    
            # Collects all taglets
            taglets = []
            for module in modules:
                taglets.extend(module.get_taglets())
            taglet_executor = TagletExecutor()
            taglet_executor.set_taglets(taglets)
    
            # Executes taglets
            log.info("Executing taglets")
            vote_matrix = taglet_executor.execute(unlabeled)

            log.info("Finished executing taglets")

            weak_labels = self._get_majority(vote_matrix)
            
            for label in weak_labels:
                unlabeled_images_labels.append(torch.FloatTensor(label))

        # Trains end model
        log.info("Training end model")

        end_model_train_data = self._combine_soft_labels(unlabeled_images_labels,
                                                         self.task.get_unlabeled_data(True),
                                                         self.task.get_labeled_train_data())
        self.end_model = EndModel(self.task)
        self.end_model.train(end_model_train_data, val)
        log.info("Finished training end model")
        return self.end_model

    def _get_taglets_modules(self):
        if self.task.scads_path is not None:
            return [PrototypeModule(task=self.task),
                    MultiTaskModule(task=self.task),
                    TransferModule(task=self.task),
                    FineTuneModule(task=self.task),
                    ZSLKGModule(task=self.task)]
        return [FineTuneModule(task=self.task),
                PrototypeModule(task=self.task)]

    def _combine_soft_labels(self, weak_labels, unlabeled_dataset, labeled_dataset):
        labeled = DataLoader(labeled_dataset, batch_size=1, shuffle=False)
        soft_labels_labeled_images = []
        for _, image_labels in labeled:
            soft_labels_labeled_images.append(torch.FloatTensor(self._to_soft_one_hot(int(image_labels[0]))))

        new_labeled_dataset = SoftLabelDataset(labeled_dataset, soft_labels_labeled_images, remove_old_labels=True)
        if unlabeled_dataset is None:
            end_model_train_data = new_labeled_dataset
        else:
            new_unlabeled_dataset = SoftLabelDataset(unlabeled_dataset, weak_labels, remove_old_labels=False)
            end_model_train_data = ConcatDataset([new_labeled_dataset, new_unlabeled_dataset])

        return end_model_train_data

    def _get_majority(self, vote_matrix):
        weak_labels = []
        for vote in vote_matrix:
            counts = np.bincount(vote)
            majority_vote = np.argmax(counts)
            weak_labels.append(self._to_soft_one_hot(majority_vote))
        return weak_labels

    def _to_soft_one_hot(self, l):
        soh = [0.1 / len(self.task.classes)] * len(self.task.classes)
        soh[l] += 0.9
        return soh
