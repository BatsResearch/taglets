from JPL_interface import JPL
from modules.module import RandomActiveLearning, TransferModule
from taglet_executer import TagletExecuter
from task import Task
import os
from pathlib import Path
import random
import pandas as pd


class Controller:
    def __init__(self):
        self.api = JPL()
        self.task = self.get_task()
        self.num_checkpoints = 3
        self.batch_size = 32
        self.num_workers = 2

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
        active_learning = RandomActiveLearning(self.task)
        examples = active_learning.find_candidates(available_budget)
        query = {'example_ids': examples}
        labeled_images = self.api.request_label(query)
        self.task.add_labeled_images(labeled_images)
        print("New labeled images:", len(labeled_images))
        print("Total labeled images:", len(self.task.labeled_images))

    def get_predictions(self):
        train_data_loader, val_data_loader, test_data_loader = self.task.load_labeled_data(self.batch_size,
                                                                                           self.num_workers)
        MNIST_module = TransferModule(task=self.task)
        MNIST_module.train_taglets(train_data_loader, val_data_loader, test_data_loader)
        taglets = MNIST_module.get_taglets()
        taglet_executer = TagletExecuter(taglets)
        label_matrix = taglet_executer.execute(self.task.get_unlabeled_images(self.batch_size,
                                                                              self.num_workers), use_gpu=False)

        # LabelModel implementation
        # soft_labels = LabelModel.annotate(label_matrix)
        # end_model = end_model(...)
        # [test_predictions] = end_model.prediction(end_model, task.evaluation_image_path)

        # Temporary implementation
        test_images = [f.name for f in Path(self.task.evaluation_image_path).iterdir() if f.is_file()]
        rand_labels = [str(random.randint(0, 10)) for _ in range(len(test_images))]
        df = pd.DataFrame({'id': test_images, 'label': rand_labels})
        predictions = df.to_dict()
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
