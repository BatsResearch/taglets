from .data import SoftLabelDataset
from .modules import FineTuneModule, PrototypeModule, TransferModule, MultiTaskModule, ZSLKGModule
from .pipeline import EndModel, TagletExecutor

import logging
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
import traceback

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
        val = self.task.get_validation_data()

        unlabeled_images_labels = []
        if unlabeled is not None:
            # Creates taglets
            modules = self._get_taglets_modules()
            taglets = []
            for cls in modules:
                try:
                    log.info("Initializing %s module", cls.__name__)
                    module = cls(task=self.task)
                    log.info("Training %s module", cls.__name__)
                    module.train_taglets(labeled, val)
                    log.info("Finished training %s module", cls.__name__)

                    # Collects taglets
                    taglets.extend(module.get_valid_taglets())
                except Exception:
                    log.error("Exception raised in %s module", cls.__name__)
                    for line in traceback.format_exc().splitlines():
                        log.error(line)
                    log.error("Continuing execution")
    
            taglet_executor = TagletExecutor()
            taglet_executor.set_taglets(taglets)
    
            # Executes taglets
            log.info("Executing taglets")
            vote_matrix1 = taglet_executor.execute(self.task.get_unlabeled_data(True))
            vote_matrix2 = taglet_executor.execute(self.task.get_unlabeled_data(False))
            log.info("Finished executing taglets")

            return vote_matrix1, vote_matrix2

        return None

    def _get_taglets_modules(self):
        if self.task.scads_path is not None:
            return [MultiTaskModule,
                    TransferModule,
                    FineTuneModule,
                    ZSLKGModule]
        return [FineTuneModule,
                PrototypeModule]

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

    def _get_weighted_dist(self, vote_matrix, weights):
        weak_labels = []
        for row in vote_matrix:
            weak_label = np.zeros((len(self.task.classes),))
            for i in range(len(row)):
                weak_label[row[i]] += weights[i]
            weak_labels.append(weak_label / weak_label.sum())
        return weak_labels

    def _to_soft_one_hot(self, l):
        soh = [0.1 / len(self.task.classes)] * len(self.task.classes)
        soh[l] += 0.9
        return soh
