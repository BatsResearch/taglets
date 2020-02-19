from .custom_dataset import SoftLabelDataSet
from .modules import FineTuneModule
from .pipeline import EndModel, TagletExecutor
import labelmodels
import logging
import numpy as np
import torch

log = logging.getLogger(__name__)


class Controller:
    def __init__(self, use_gpu=False):
        self.task = None
        self.end_model = None
        self.taglet_executor = None
        self.batch_size = 32
        self.num_workers = 2
        self.use_gpu = use_gpu

    def train_end_model(self, task):
        """
        Executes a training pipeline end-to-end, turning a Task into an EndModel
        :param task: description of the task for the EndModel
        :return: A trained EndModel
        """
        self.task = task

        # Creates data loaders
        train_data_loader, val_data_loader,  train_image_names, train_image_labels = self.task.load_labeled_data(
            self.batch_size,
            self.num_workers)

        unlabeled_data_loader, unlabeled_image_names = self.task.load_unlabeled_data(self.batch_size,
                                                                                     self.num_workers)

        # Initializes taglet-creating modules
        modules = self._get_taglets_modules()
        for module in modules:
            log.info("Training %s module", module.__class__.__name__)
            module.train_taglets(train_data_loader, val_data_loader, self.use_gpu)
            log.info("Finished training %s module", module.__class__.__name__)

        # Collects all taglets
        taglets = []
        for module in modules:
            taglets.extend(module.get_taglets())
        self.taglet_executor = TagletExecutor()
        self.taglet_executor.set_taglets(taglets)

        # Executes taglets
        log.info("Executing taglets")
        vote_matrix = self.taglet_executor.execute(unlabeled_data_loader, self.use_gpu)
        log.info("Finished executing taglets")

        # Learns label model
        log.info("Training label model")
        labelmodel = labelmodels.NaiveBayes(
            num_classes=len(task.classes), num_lfs=len(taglets))
        labelmodel.estimate_label_model(vote_matrix)
        log.info("Finished training label model")

        # Computes label distribution
        log.info("Getting label distribution")
        weak_labels = labelmodel.get_label_distribution(vote_matrix)
        log.info("Finished getting label distribution")


        # Trains end model
        log.info("Training end model")
        self.end_model = EndModel(self.task)

        end_model_train_data_loader = self.combine_soft_labels(soft_labels_unlabeled_images,
                                                               unlabeled_image_names,
                                                               train_image_names, train_image_labels)
        self.end_model.train(end_model_train_data_loader, val_data_loader, self.use_gpu)
        log.info("Finished training end model")

        return self.end_model

    def _get_taglets_modules(self):
        return [FineTuneModule(task=self.task)]

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
