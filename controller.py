from JPL_interface import JPL
from modules.module import TransferModule
from modules.active_learning import RandomActiveLearning, LeastConfidenceActiveLearning
from taglets.taglet_executer import TagletExecutor
from task import Task
from pathlib import Path
import random
import pandas as pd
from label_model import get_label_distribution
from custom_dataset import CustomDataSet, SoftLabelDataSet
import torch
from taglets.taglet import EndModel
import numpy as np



class Controller:
    def __init__(self):
        self.api = JPL()
        self.task = self.get_task()
        self.active_learning = RandomActiveLearning(self.task)
        # self.active_learning = LeastConfidenceActiveLearning(self.task)
        self.num_checkpoints = 1
        self.batch_size = 32
        self.num_workers = 2
        self.use_gpu = False
        self.endmodel = EndModel(self.task)

    def run_checkpoints(self):
        for i in range(self.num_checkpoints):
            self.request_labels()
            predictions = self.get_predictions()
            self.submit_predictions(predictions)

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

    def request_labels(self):
        session_status = self.api.get_session_status()
        available_budget = session_status['budget_left_until_checkpoint']
        available_budget = available_budget // 10   # For testing
        examples = self.active_learning.find_candidates(available_budget, use_gpu=self.use_gpu)
        query = {'example_ids': examples}
        labeled_images = self.api.request_label(query)
        self.task.add_labeled_images(labeled_images)
        print("New labeled images:", len(labeled_images))
        print("Total labeled images:", len(self.task.labeled_images))

    def to_soft_one_hot(self,l):
        soh = [0.15] * len(self.task.classes)
        soh[l] = 0.85
        return soh

    def get_predictions(self):
        train_data_loader, val_data_loader, test_data_loader, image_names, image_labels= self.task.load_labeled_data(self.batch_size,
                                                                                           self.num_workers)
        unlabeled_data_loader , unlabeled_images_names= self.task.load_unlabeled_data(self.batch_size, self.num_workers)
        mnist_module = TransferModule(task=self.task)

        print("Training taglets on labeled data...")
        mnist_module.train_taglets(train_data_loader, val_data_loader, test_data_loader, self.use_gpu)
        taglets = mnist_module.get_taglets()
        taglet_executor = TagletExecutor(taglets)

        print("Executing taglets on unlabled data...")
        label_matrix = taglet_executor.execute(unlabeled_data_loader, self.use_gpu)

        # LabelModel
        print("Label Model...")
        soft_lafbels_unlabeled_images = get_label_distribution(label_matrix, len(self.task.classes))

        # End Model
        soft_labels_labeled_images = []
        for i in range(len(image_labels)):
            soft_labels_labeled_images.append(self.to_soft_one_hot(int(image_labels[i])))

        soft_labels_labeled_images = np.array(soft_labels_labeled_images)
        all_soft_labels = np.concatenate((soft_lafbels_unlabeled_images,soft_labels_labeled_images), axis=0)

        all_images = unlabeled_images_names + image_names

        endmodel_data = SoftLabelDataSet(self.task.unlabeled_image_path,
                                            all_images,
                                            all_soft_labels,
                                            self.task._transform_image(),
                                            self.task.number_of_channels)


        endmodel_data_loader = torch.utils.data.DataLoader(endmodel_data,
                                                            batch_size= self.batch_size,
                                                            shuffle=True,
                                                            num_workers=self.num_workers)


        self.endmodel.train(endmodel_data_loader, None, None, self.use_gpu)

        predictions =  self.endmodel.predict(self.task.evaluation_image_path,self.task.number_of_channels, self.task._transform_image(),self.use_gpu)



        # Temporary implementation
        # test_images = [f.name for f in Path(self.task.evaluation_image_path).iterdir() if f.is_file()]
        # rand_labels = [str(random.randint(0, 10)) for _ in range(len(test_images))]
        # df = pd.DataFrame({'id': test_images, 'label': rand_labels})
        # predictions = df.to_dict()
        # return predictions

        return predictions

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
