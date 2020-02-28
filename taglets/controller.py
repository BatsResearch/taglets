from taglets.data.custom_dataset import SoftLabelDataSet
from .modules import FineTuneModule
from .pipeline import EndModel, TagletExecutor

import labelmodels
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)


class Controller:
    """
    Manages training and execution of taglets, as well as training EndModels
    """
    def __init__(self, task, batch_size=32, num_workers=2, use_gpu=False):
        self.task = task
        self.end_model = None
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_gpu = use_gpu

    def train_end_model(self):
        """
        Executes a training pipeline end-to-end, turning a Task into an EndModel

        :param task: description of the task for the EndModel
        :return: A trained EndModel
        """

        # Creates data loaders
        labeled_dataset = self.task.get_labeled_train_data()
        unlabeled_dataset = self.task.get_unlabeled_train_data()
        val_dataset = self.task.get_validation_data()
        labeled = self._get_data_loader(labeled_dataset, shuffle=True)
        unlabeled = self._get_data_loader(unlabeled_dataset, shuffle=True)
        val = self._get_data_loader(val_dataset, shuffle=False)

        # Initializes taglet-creating modules
        modules = self._get_taglets_modules()
        for module in modules:
            log.info("Training %s module", module.__class__.__name__)
            module.train_taglets(labeled, val, self.use_gpu)
            log.info("Finished training %s module", module.__class__.__name__)

        # Collects all taglets
        taglets = []
        for module in modules:
            taglets.extend(module.get_taglets())
        taglet_executor = TagletExecutor()
        taglet_executor.set_taglets(taglets)

        # Executes taglets
        log.info("Executing taglets")
        vote_matrix = taglet_executor.execute(unlabeled, self.use_gpu)
        log.info("Finished executing taglets")

        # Learns label model
        labelmodel = self._train_label_model(vote_matrix)

        # Computes label distribution
        log.info("Getting label distribution")
        weak_labels = labelmodel.get_label_distribution(vote_matrix)
        log.info("Finished getting label distribution")

        # Trains end model
        log.info("Training end model")
        self.end_model = EndModel(self.task)

        train_image_names = labeled_dataset.get_filenames()
        train_image_labels = labeled_dataset.get_labels()
        unlabeled_image_names = []
        for batch in unlabeled:
            for names in batch[2]:
                unlabeled_image_names.append(names)
        end_model_train_data_loader = self.combine_soft_labels(weak_labels,
                                                               unlabeled_image_names,
                                                               train_image_names, train_image_labels)
        self.end_model.train(end_model_train_data_loader, val, self.use_gpu)
        log.info("Finished training end model")

        return self.end_model

    def _get_taglets_modules(self):
        return [FineTuneModule(task=self.task)]

    def _get_data_loader(self, dataset, shuffle=True):
        """
        Creates a DataLoader for the given dataset

        :param dataset: Dataset to wrap
        :param shuffle: whether to shuffle the data
        :return: the DataLoader
        """
        return DataLoader(dataset, batch_size=self.batch_size,
                          shuffle=shuffle, num_workers=self.num_workers)

    def _train_label_model(self, vote_matrix):
        log.info("Training label model")
        labelmodel = labelmodels.NaiveBayes(
            num_classes=vote_matrix.shape[0], num_lfs=vote_matrix.shape[1])
        labelmodel.estimate_label_model(vote_matrix)
        log.info("Finished training label model")
        return labelmodel

    def combine_soft_labels(self, unlabeled_labels, unlabeled_names, train_image_names, train_image_labels):
        def to_soft_one_hot(l):
            soh = [0.15] * len(self.task.classes)
            soh[l] = 0.85
            return soh

        soft_labels_labeled_images = []
        for image_label in train_image_labels:
            soft_labels_labeled_images.append(to_soft_one_hot(int(image_label)))

        all_soft_labels = np.concatenate((unlabeled_labels, np.array(soft_labels_labeled_images)), axis=0)
        all_names = unlabeled_names + train_image_names

        end_model_train_data = SoftLabelDataSet(self.task.unlabeled_image_path,
                                                all_names,
                                                all_soft_labels,
                                                self.task.transform_image(),
                                                self.task.number_of_channels)

        train_data = torch.utils.data.DataLoader(end_model_train_data,
                                                 batch_size=self.batch_size,
                                                 shuffle=True,
                                                 num_workers=self.num_workers)

        return train_data
