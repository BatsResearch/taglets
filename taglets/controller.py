import gc
from .data import SoftLabelDataset
from .modules import FineTuneModule, PrototypeModule, TransferModule, MultiTaskModule
from .pipeline import EndModel, TagletExecutor

import labelmodels
import logging
import torch
from torch.utils.data import DataLoader, ConcatDataset

from memory_profiler import profile
from pympler import muppy, tracker, asizeof, summary
import linecache
import os
import tracemalloc
import psutil

log = logging.getLogger(__name__)


def display_top(snapshot, key_type='lineno', limit=10):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)
    
    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        print("#%s: %s:%s: %.1f KiB"
              % (index, frame.filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))


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

    @profile
    def train_end_model(self):
        """
        Executes a training pipeline end-to-end, turning a Task into an EndModel

        :param task: description of the task for the EndModel
        :return: A trained EndModel
        """

        # Add to leaky code within python_script_being_profiled.py
        tracemalloc.start()

        print("DEBUG - Psutil: START of train_end_model")
        print(psutil.virtual_memory())

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
            # plus 1 because labelmodel 1-based indexing (0 is for restraining from voting)
            vote_matrix += 1
            log.info("Finished executing taglets")

            print("DEBUG - Tracemalloc: Before getting weak labels")
            snapshot = tracemalloc.take_snapshot()
            display_top(snapshot)

            print("DEBUG - Psutil: Before getting weak labels")
            print(psutil.virtual_memory())

            tr = tracker.SummaryTracker()
    
            # Learns label model
            labelmodel = self._train_label_model(vote_matrix)

            print("DEBUG - Pympler tracker: Memory diff before and after getting weak labels")
            tr.print_diff()

            print("DEBUG - Pympler asizeof: size of labelmodel")
            print(asizeof.asizeof(labelmodel))

            print("DEBUG - Pympler summary: After getting weak labels")
            all_objects = muppy.get_objects()
            sum1 = summary.summarize(all_objects)
            summary.print_(sum1)

            print("DEBUG - Psutil: After getting weak labels")
            print(psutil.virtual_memory())

            print("DEBUG - Tracemalloc: After getting weak labels")
            snapshot = tracemalloc.take_snapshot()
            display_top(snapshot)
            
            # Computes label distribution
            print("Getting label distribution")
            weak_labels = labelmodel.get_label_distribution(vote_matrix)
            print("Finished getting label distribution")
            
            
            del labelmodel
            gc.collect()
            
            for label in weak_labels:
                unlabeled_images_labels.append(torch.FloatTensor(label))

        # Trains end model
        log.info("Training end model")

        end_model_train_data_loader = self.combine_soft_labels(unlabeled_images_labels,
                                                               self.task.get_unlabeled_train_data(),
                                                               self.task.get_labeled_train_data())
        self.end_model = EndModel(self.task)
        
        self.end_model.train(end_model_train_data_loader, val, self.use_gpu)
        log.info("Finished training end model")

        print("DEBUG - Psutil: END of train_end_model")
        print(psutil.virtual_memory())

        return self.end_model

    def _get_taglets_modules(self):
        if self.task.scads_path:
            return [FineTuneModule(task=self.task), PrototypeModule(task=self.task), TransferModule(task=self.task), MultiTaskModule(task=self.task)]
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
        config = labelmodels.LearningConfig()
        config.epochs = 5
        config.batch_size = 8
        labelmodel = labelmodels.NaiveBayes(
            num_classes=len(self.task.classes), num_lfs=vote_matrix.shape[1])
        labelmodel.estimate_label_model(vote_matrix, config=config)
        log.info("Finished training label model")
        return labelmodel

    def combine_soft_labels(self, weak_labels, unlabeled_dataset, labeled_dataset):
        def to_soft_one_hot(l):
            soh = [0.1 / len(self.task.classes)] * len(self.task.classes)
            soh[l] += 0.9
            return soh

        labeled = DataLoader(labeled_dataset, batch_size=1, shuffle=False)
        soft_labels_labeled_images = []
        for _, image_labels in labeled:
            soft_labels_labeled_images.append(torch.FloatTensor(to_soft_one_hot(int(image_labels[0]))))

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
