import os
import logging
from logging import StreamHandler
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
import traceback
from accelerate import Accelerator
accelerator = Accelerator()

from .data import SoftLabelDataset
from .modules import FineTuneModule, TransferModule, MultiTaskModule, ZSLKGModule, FixMatchModule, NaiveVideoModule, \
    RandomModule, DannModule, BaselineVideoModule
from .pipeline import ImageEndModel, VideoEndModel, RandomEndModel, TagletExecutor

####################################################################
# We configure logging in the main class of the application so that
# subprocesses inherit the same configuration. This would have to be
# redesigned if used as part of a larger application with its own
# logging configuration
####################################################################

logger_ = logging.getLogger()
logger_.level = logging.INFO
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')


class AccelerateHandler(StreamHandler):
    def __init__(self, stream):
        super().__init__(stream)
    
    def emit(self, record):
        if accelerator.is_local_main_process:
            super().emit(record)


stream_handler = AccelerateHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)
logger_.addHandler(stream_handler)

# if not os.environ.get("CI"):
#     import logger
    
#     class JPLHandler(StreamHandler):
#         "Handle the log stream and wrap it into the JPL logger."
#         def __init__(self):
#             StreamHandler.__init__(self)

#         def emit(self, record):
#             if accelerator.is_local_main_process:
#                 msg = self.format(record)
#                 self.jpl_logger = logger.log(msg, 'Brown', 0) # For the moment fixed checkpoint


    # jpl_handler = JPLHandler()
    # jpl_handler.setLevel(logging.INFO)
    # jpl_handler.setFormatter(formatter)
    # logger_.addHandler(jpl_handler)

####################################################################
# End of logging configuration
####################################################################

log = logging.getLogger(__name__)


class Controller:
    """
    Manages training and execution of taglets, as well as training EndModels
    """
    def __init__(self, task, simple_run=False):
        self.task = task
        self.end_model = None
        self.simple_run = simple_run

    def train_end_model(self):
        """
        Executes a training pipeline end-to-end, turning a Task into an EndModel
        :return: A trained EndModel
        """
        # Gets datasets
        labeled = self.task.get_labeled_train_data()
        unlabeled_test = self.task.get_unlabeled_data(False) # augmentation is not applied
        unlabeled_train = self.task.get_unlabeled_data(True) # augmentation is applied
        val = self.task.get_validation_data()

        unlabeled_images_labels = []
        if unlabeled_test is not None:
            # Creates taglets
            modules = self._get_taglets_modules()
            taglets = []
            for cls in modules:
                try:
                    log.info("Initializing %s module", cls.__name__)
                    module = cls(task=self.task)
                    log.info("Training %s module", cls.__name__)
                    module.train_taglets(labeled, val, unlabeled_train)
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
            #log.info(f"FILEPATHS TEST {unlabeled_test.filepaths}")
            vote_matrix = taglet_executor.execute(unlabeled_test)
            log.info("Finished executing taglets")
            
            if self.task.unlabeled_train_labels is not None:
                log.info('Accuracies of each taglet on the unlabeled train data:')
                for i in range(len(taglets)):
                    acc = np.sum(vote_matrix[:, i] == self.task.unlabeled_train_labels) / len(self.task.unlabeled_train_labels)
                    log.info("Module {} - acc {:.4f}".format(taglets[i].name, acc))

            # Combines taglets' votes into soft labels
            if val is not None and len(val) >= len(self.task.classes) * 10:
                # Weight votes using development set
                weights = [taglet.evaluate(val) for taglet in taglets]
                log.info("Validation accuracies of each taglet:")
                for w, taglet in zip(weights, taglets):
                    log.info("Module {} - acc {:.4f}".format(taglet.name, w))
            else:
                # Weight all votes equally
                weights = [1.0] * len(taglets)

            weak_labels = self._get_weighted_dist(vote_matrix, weights)
            
            if self.task.unlabeled_train_labels is not None:
                log.info('Accuracy of the labelmodel on the unlabeled train data:')
                predictions = np.asarray([np.argmax(label) for label in weak_labels])
                acc = np.sum(predictions == self.task.unlabeled_train_labels) / len(self.task.unlabeled_train_labels)
                log.info('Acc {:.4f}'.format(acc))
            
            for label in weak_labels:
                unlabeled_images_labels.append(torch.FloatTensor(label))

        # Trains end model
        log.info("Training end model")

        end_model_train_data = self._combine_soft_labels(unlabeled_images_labels,
                                                         unlabeled_train,
                                                         self.task.get_labeled_train_data())
        if self.simple_run:
            self.end_model = RandomEndModel(self.task)
        elif self.task.video_classification:
            self.end_model = VideoEndModel(self.task)
        else:
            self.end_model = ImageEndModel(self.task)
        self.end_model.train(end_model_train_data, val)
        log.info("Finished training end model")

        if self.task.unlabeled_train_labels is not None and unlabeled_test is not None:
            log.info('Accuracy of the end model on the unlabeled train data:')
            outputs = self.end_model.predict(unlabeled_test)
            predictions = np.argmax(outputs, 1)
            acc = np.sum(predictions == self.task.unlabeled_train_labels) / len(self.task.unlabeled_train_labels)
            log.info('Acc {:.4f}'.format(acc))
        
        return self.end_model

    def _get_taglets_modules(self):
        if self.simple_run:
             return [RandomModule]
        elif self.task.video_classification:
             return [BaselineVideoModule]#NaiveVideoModule]
        else:
            if self.task.scads_path is not None:
                return [DannModule, 
                        MultiTaskModule,
                        ZSLKGModule,
                        TransferModule,
                        FineTuneModule,
                        FixMatchModule]
            return [FineTuneModule, FixMatchModule]

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
