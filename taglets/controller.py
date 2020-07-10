
from .data import SoftLabelDataset
from .modules import FineTuneModule, PrototypeModule, TransferModule, MultiTaskModule
from .pipeline import EndModel, TagletExecutor

import labelmodels
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset

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
        labeled = self._get_data_loader(self.task.get_labeled_train_data(), shuffle=True)
        unlabeled = self._get_data_loader(self.task.get_unlabeled_train_data(), shuffle=False)
        val = self._get_data_loader(self.task.get_validation_data(), shuffle=False)

        unlabeled_images_labels = []
        if unlabeled is not None:
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
            # # plus 1 because labelmodel 1-based indexing (0 is for restraining from voting)
            # vote_matrix += 1
            log.info("Finished executing taglets")
    
            # # Learns label model
            # labelmodel = self._train_label_model(vote_matrix)
            #
            # # Computes label distribution
            # log.info("Getting label distribution")
            # weak_labels = labelmodel.get_label_distribution(vote_matrix)
            # log.info("Finished getting label distribution")
            
            weak_labels = self._get_majority(vote_matrix)
            
            for label in weak_labels:
                unlabeled_images_labels.append(torch.FloatTensor(label))

        # Trains end model
        log.info("Training end model")

        end_model_train_data_loader = self._combine_soft_labels(unlabeled_images_labels,
                                                               self.task.get_unlabeled_train_data(),
                                                               self.task.get_labeled_train_data())
        self.end_model = EndModel(self.task)
        self.end_model.train(end_model_train_data_loader, val, self.use_gpu)
        log.info("Finished training end model")

        return self.end_model

    def _get_taglets_modules(self):
        if self.task.scads_path:
            return [ TransferModule(task=self.task),FineTuneModule(task=self.task)]
            # return [FineTuneModule(task=self.task), PrototypeModule(task=self.task), TransferModule(task=self.task), MultiTaskModule(task=self.task)]
        return [FineTuneModule(task=self.task), PrototypeModule(task=self.task)]

    def _get_data_loader(self, dataset, shuffle=True):
        """
        Creates a DataLoader for the given dataset

        :param dataset: Dataset to wrap
        :param shuffle: whether to shuffle the data
        :return: the DataLoader
        """
        if dataset != None:
            return DataLoader(dataset, batch_size=self.batch_size,
                            shuffle=shuffle, num_workers=self.num_workers)
        else:
            return None

    def _train_label_model(self, vote_matrix):
        log.info("Training label model")
        labelmodel = labelmodels.NaiveBayes(
            num_classes=len(self.task.classes), num_lfs=vote_matrix.shape[1])
        labelmodel.estimate_label_model(vote_matrix)
        log.info("Finished training label model")
        return labelmodel

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
    
        train_data = torch.utils.data.DataLoader(end_model_train_data,
                                                 batch_size=self.batch_size,
                                                 shuffle=True,
                                                 num_workers=0)
    
        return train_data

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
