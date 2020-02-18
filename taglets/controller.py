from .modules import FineTuneModule
from .pipeline import TagletExecutor
from .custom_dataset import SoftLabelDataSet
import torch
from .pipeline import EndModel
import numpy as np
import datetime


class Controller:
    def __init__(self, task, use_gpu=False, testing=False):
        self.taglet_executor = TagletExecutor()
        self.end_model = EndModel(task)
        self.task = task
        self.batch_size = 32
        self.num_workers = 2
        self.use_gpu = use_gpu
        self.testing = testing

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

    def get_predictions(self, phase):
        """train taglets, label model, and endmodel, and return prediction
        :param phase: 'base' or 'adapt'
        """
        train_data_loader, val_data_loader,  train_image_names, train_image_labels = self.task.load_labeled_data(
            self.batch_size,
            self.num_workers)

        unlabeled_data_loader, unlabeled_image_names = self.task.load_unlabeled_data(self.batch_size,
                                                                                     self.num_workers)

        fine_tune_module = FineTuneModule(task=self.task)
        modules = [fine_tune_module]

        print("**********Training taglets on labeled data**********")
        t1 = datetime.datetime.now()
        fine_tune_module.train_taglets(train_data_loader, val_data_loader, self.use_gpu, self.testing)
        t2 = datetime.datetime.now()
        print()
        print(".....Taglet training time: {}".format((t2 - t1).seconds))

        taglets = []
        for module in modules:
            taglets.extend(module.get_taglets())
        self.taglet_executor.set_taglets(taglets)

        print("**********Executing taglets on unlabled data**********")
        t1 = datetime.datetime.now()
        label_matrix, candidates = self.taglet_executor.execute(unlabeled_data_loader, self.use_gpu, self.testing)
        self.confidence_active_learning.set_candidates(candidates)
        t2 = datetime.datetime.now()
        print()
        print(".....Taglet executing time: {}".format((t2 - t1).seconds))


        print("**********Label Model**********")
        t1 = datetime.datetime.now()
        soft_labels_unlabeled_images = get_label_distribution(label_matrix, len(self.task.classes), self.testing)
        t2 = datetime.datetime.now()
        print()
        print(".....Label Model time: {}".format((t2 - t1).seconds))



        print("**********End Model**********")
        t1 = datetime.datetime.now()
        if self.testing:
            unlabeled_image_names = unlabeled_image_names[:len(soft_labels_unlabeled_images)]
        end_model_train_data_loader = self.combine_soft_labels(soft_labels_unlabeled_images,
                                                         unlabeled_image_names,
                                                         train_image_names, train_image_labels)
        self.end_model.train(end_model_train_data_loader, val_data_loader, self.use_gpu, self.testing)
        t2 = datetime.datetime.now()
        print()
        print(".....End Model time: {}".format((t2 - t1).seconds))

        return self.end_model.predict(self.task.evaluation_image_path,
                                      self.task.number_of_channels,
                                      self.task.transform_image(),
                                      self.use_gpu)
