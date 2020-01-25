from JPL_interface import JPL
from modules.module import TransferModule
from modules.active_learning import RandomActiveLearning, LeastConfidenceActiveLearning
from taglets.taglet_executer import TagletExecutor
from task import Task
from label_model import get_label_distribution
from custom_dataset import CustomDataSet, SoftLabelDataSet
import torch
from taglets.end_model import EndModel
import numpy as np


class Controller:
    def __init__(self):
        self.api = JPL()
        self.task = self.get_task()
        self.random_active_learning = RandomActiveLearning()
        self.confidence_active_learning = LeastConfidenceActiveLearning()
        self.taglet_executor = TagletExecutor()
        self.end_model = EndModel(self.task)
        self.num_base_checkpoints = 3
        self.num_adapt_checkpoints = 3
        self.batch_size = 32
        self.num_workers = 2
        self.use_gpu = False

    def run_checkpoints(self):
        for i in range(self.num_base_checkpoints):
            print('---base check point: {}'.format(i))
            available_budget = self.get_available_budget()
            unlabeled_image_names = self.task.get_unlabeled_image_names()
            print('number of unlabeled data: {}'.format(len(unlabeled_image_names)))
            if i == 0:
                candidates = self.random_active_learning.find_candidates(available_budget, unlabeled_image_names)
            else:
                candidates = self.confidence_active_learning.find_candidates(available_budget, unlabeled_image_names)
            self.request_labels(candidates)
            predictions = self.get_predictions()
            print(predictions)
            # self.submit_predictions(predictions)

    def get_task(self):
        task_names = self.api.get_available_tasks()
        task_name = task_names[1]  # Image classification task
        self.api.create_session(task_name)
        task_metadata = self.api.get_task_metadata(task_name)

        task = Task(task_metadata)
        session_status = self.api.get_session_status()
        current_dataset = session_status['current_dataset']
        task.classes = current_dataset['classes']
        task.number_of_channels = current_dataset['number_of_channels']

        task.unlabeled_image_path = "./sql_data/MNIST/train"
        task.evaluation_image_path = "./sql_data/MNIST/test"  # Should be updated later
        task.labeled_images = self.api.get_seed_labels()
        return task

    def get_available_budget(self):
        session_status = self.api.get_session_status()
        available_budget = session_status['budget_left_until_checkpoint']
        available_budget = available_budget // 10   # For testing
        return available_budget

    def request_labels(self, examples):
        query = {'example_ids': examples}
        labeled_images = self.api.request_label(query)
        self.task.add_labeled_images(labeled_images)
        print("New labeled images:", len(labeled_images))
        print("Total labeled images:", len(self.task.labeled_images))

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

        print('^^^^')
        print(len(train_data.dataset))

        return train_data




    def get_predictions(self):
        train_data_loader, val_data_loader,  train_image_names, train_image_labels=\
            self.task.load_labeled_data(
            self.batch_size,
            self.num_workers)
        unlabeled_data_loader, unlabeled_image_names = self.task.load_unlabeled_data(self.batch_size,
                                                                                     self.num_workers)

        mnist_module = TransferModule(task=self.task)

        print("Training taglets on labeled data...")
        mnist_module.train_taglets(train_data_loader, val_data_loader, self.use_gpu)
        taglets = mnist_module.get_taglets()
        self.taglet_executor.set_taglets(taglets)

        print("Executing taglets on unlabled data...")
        label_matrix, candidates = self.taglet_executor.execute(unlabeled_data_loader, self.use_gpu)
        self.confidence_active_learning.set_candidates(candidates)

        print("Label Model...")
        soft_labels_unlabeled_images = get_label_distribution(label_matrix, len(self.task.classes))

        print("End Model...")
        end_model_train_data_loader = self.combine_soft_labels(soft_labels_unlabeled_images,
                                                         unlabeled_image_names,
                                                         train_image_names, train_image_labels)
        self.end_model.train(end_model_train_data_loader, val_data_loader, self.use_gpu)
        return self.end_model.predict(self.task.evaluation_image_path,
                                      self.task.number_of_channels,
                                      self.task.transform_image(),
                                      self.use_gpu)

    def submit_predictions(self, predictions):
        self.api.submit_prediction(predictions)
        session_status = self.api.get_session_status()
        print("Checkpoint scores", session_status['checkpoint_scores'])
        print("Phase:", session_status['pair_stage'])


def main():
    controller = Controller()
    controller.run_checkpoints()


if __name__ == "__main__":
    main()
