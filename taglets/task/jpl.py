import argparse
import logging
import sys
import requests
import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils import data
from ..data import CustomDataset
from ..active import RandomActiveLearning, LeastConfidenceActiveLearning
from ..task import Task
from ..controller import Controller
from ..models import MnistResNet

log = logging.getLogger(__name__)


class JPL:
    """
    A class to interact with JPL-like APIs.
    """
    def __init__(self):
        """
        Create a new JPL object.
        """

        self.secret = 'a5aed2a8-db80-4b22-bf72-11f2d0765572'
        self.url = 'https://api-staging.lollllz.com'
        self.session_token = ''
        self.data_type = 'sample'   # Sample or full

    def get_available_tasks(self, problem_type):
        """
        Get all available tasks.
        :return: A list of tasks (problems)
        """
        headers = {'user_secret': self.secret}
        r = requests.get(self.url + "/list_tasks", headers=headers)
        task_list = r.json()['tasks']

        subset_tasks = []
        for _task in task_list:
            r = requests.get(self.url+"/task_metadata/"+_task, headers=headers)
            task_metadata = r.json()
            if task_metadata['task_metadata']['problem_type'] == problem_type:
                subset_tasks.append(_task)
        return subset_tasks

    def get_task_metadata(self, task_name):
        """
        Get metadata about a task.
        :param task_name: The name of the task (problem)
        :return: The task metadata
        """
        headers = {'user_secret': self.secret}
        r = requests.get(self.url + "/task_metadata/" + task_name, headers=headers)
        return r.json()['task_metadata']

    def create_session(self, task_name):
        """
        Create a new session.
        :param task_name: The name of the task (problem
        :return: None
        """
        headers = {'user_secret': self.secret}
        # r = requests.get(self.url + "/auth/get_session_token/" + self.data_type + "/" + task_name, headers=headers)
        r = requests.post(self.url + "/auth/create_session",
                          json={'session_name': 'testing', 'data_type': self.data_type, 'task_id': task_name},
                          headers=headers)

        session_token = r.json()['session_token']
        self.session_token = session_token

    def get_session_status(self):
        """
        Get the session status.
        :return: The session status
        """
        headers = {'user_secret': self.secret, 'session_token': self.session_token}
        r = requests.get(self.url + "/session_status", headers=headers)
        if 'Session_Status' in r.json():
            return r.json()['Session_Status']
        else:
            return {}

    def get_initial_seed_labels(self):
        """
        Get seed labels.
        :return: A list of lists with name and label e.g., ['1.png', '2'], ['2.png', '7'], etc.
        """
        headers = {'user_secret': self.secret, 'session_token': self.session_token}
        r = requests.get(self.url + "/seed_labels", headers=headers)
        labels = r.json()['Labels']
        seed_labels = []
        for image in labels:
            # Elaheh: changed based on the latest version of API
            seed_labels.append([image["class"], image["id"]])
            # seed_labels.append([image["id"], image["class"]])
        return seed_labels

    def get_secondary_seed_labels(self):
        headers = {'user_secret': self.secret, 'session_token': self.session_token}
        r = requests.get(self.url + "/secondary_seed_labels", headers=headers)
        labels = r.json()['Labels']
        secondary_seed_labels = []
        for image in labels:
            secondary_seed_labels.append([image["class"], image["id"]])
        return secondary_seed_labels


    def request_label(self, query):
        """
        Get labels of requested examples.

        :param query: A dictionary where the key is 'example_ids', and the value is a list of filenames.
        For example:
        query = {
            'example_ids': [
                '45781.png',
                '40214.png',
                '49851.png',
            ]
        }

        :return: A list of lists containing the name of the example and its label.
        For example:
         [['45781.png', '3'], ['40214.png', '7'], ['49851.png', '8']]
        """
        headers = {'user_secret': self.secret, 'session_token': self.session_token}
        r = requests.post(self.url + "/query_labels", json=query, headers=headers)
        return r.json()['Labels']

    def submit_prediction(self, predictions):
        """
        Submit predictions on test images.

        :param predictions: A dictionary containing test image names and corresponding labels.
        For example:
        predictions = {'id': ['6831.png', '1186.png', '8149.png', '4773.png', '3752.png'],
                       'label': ['9', '6', '9', '2', '10']}
        :return: The session status after submitting prediction
        """

        headers = {'user_secret': self.secret, 'session_token': self.session_token}
        r = requests.post(self.url + "/submit_predictions", json={'predictions': predictions}, headers=headers)
        return r.json()
    
    
class JPLStorage:
    def __init__(self, task_name, metadata):
        """
        Create a new Task.
        :param metadata: The metadata of the Task.
        """
        self.name = task_name
        self.description = ''
        self.problem_type = metadata['problem_type']
        self.task_id = metadata['task_id']
        self.classes = []
        self.evaluation_image_path = "path to test images"
        self.unlabeled_image_path = "path to unlabeled images"
        self.labeled_images = []  # A list of tuples with name and label e.g., ['1.png', '2'], ['2.png', '7'], etc.
        self.number_of_channels = None
        self.train_data_loader = None
        self.phase = None  # base or adaptation
        self.pretrained = None  # can load from pretrained models on ImageNet
    
    def add_labeled_images(self, new_labeled_images):
        """
        Add new labeled images to the Task.
        :param new_labeled_images: A list of lists containing the name of an image and their labels
        :return: None
        """
        self.labeled_images.extend(new_labeled_images)
        
    def set_image_path(self, dataset_dir, data_type):
        """
        Set self.evaluation_image_path and self.unlabeled_image_path with the given dataset_dir
        :param dataset_dir: the directory to the dataset
        :param data_type: 'sample' or 'full'
        :return:
        """
        self.unlabeled_image_path = os.path.join(dataset_dir,
                                                 os.path.basename(dataset_dir) + "_" + data_type,
                                                 "train")
        self.evaluation_image_path = os.path.join(dataset_dir,
                                                  os.path.basename(dataset_dir) + "_" + data_type,
                                                  "test")

    def transform_image(self):
        """
        Get the transform to be used on an image.
        :return: A transform
        """
        data_mean = [0.485, 0.456, 0.406]
        data_std = [0.229, 0.224, 0.225]
        
        return transforms.Compose([
            # transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=data_mean, std=data_std)
        ])
    
    def get_labeled_images_list(self):
        """get list of image names and labels"""
        # Elaheh: changed the following line
        image_names = [item[1] for item in self.labeled_images]
        image_labels = [item[0] for item in self.labeled_images]

        return image_names, image_labels
    
    def get_unlabeled_image_names(self):
        """return list of name of unlabeled images"""
        labeled_image_names = {img_name for img_name, label in self.labeled_images}
        unlabeled_image_names = []
        for img in os.listdir(self.unlabeled_image_path):
            if img not in labeled_image_names:
                unlabeled_image_names.append(img)
        return unlabeled_image_names
    
    def get_evaluation_image_names(self):
        evaluation_image_names = []
        for img in os.listdir(self.evaluation_image_path):
            evaluation_image_names.append(img)
        return evaluation_image_names

    def get_labeled_dataset(self):
        """
        Get training, validation, and testing data loaders from labeled data.
        :return: Training, validation, and testing data loaders
        """
        transform = self.transform_image()
    
        image_names, image_labels = self.get_labeled_images_list()

        image_paths = [os.path.join(self.unlabeled_image_path, image_name) for image_name in image_names]

        train_val_data = CustomDataset(image_paths,
                                       image_labels,
                                       transform)
    
        # 80% for training, 20% for validation
        train_percent = 0.8
        num_data = len(train_val_data)
        indices = list(range(num_data))
        train_split = int(np.floor(train_percent * num_data))
        np.random.shuffle(indices)
        train_idx = indices[:train_split]
        valid_idx = indices[train_split:]
    
        train_dataset = data.Subset(train_val_data, train_idx)
        val_dataset = data.Subset(train_val_data, valid_idx)
    
        return train_dataset, val_dataset

    def get_unlabeled_dataset(self):
        """
        Get a data loader from unlabeled data.
        :return: A data loader containing unlabeled data
        """
        transform = self.transform_image()
    
        image_names = self.get_unlabeled_image_names()
        image_paths = [os.path.join(self.unlabeled_image_path, image_name) for image_name in image_names]
        return CustomDataset(image_paths,
                             None,
                             transform)
    
    def get_evaluation_dataset(self):
        """
        Get a data loader from evaluation/test data.
        :return: A data loader containing unlabeled data
        """
        transform = self.transform_image()

        evaluation_image_names = []
        for img in os.listdir(self.evaluation_image_path):
            evaluation_image_names.append(img)
        image_paths = [os.path.join(self.evaluation_image_path, image_name) for image_name in evaluation_image_names]
        return CustomDataset(image_paths,
                             None,
                             transform)


class JPLRunner:
    def __init__(self, base_dataset_dir, adapt_dataset_dir, batch_size=32, num_workers=2,
                 use_gpu=False, testing=False, data_type='sample'):
        self.base_dataset_dir = base_dataset_dir
        self.adapt_dataset_dir = adapt_dataset_dir
        
        self.api = JPL()
        self.api.data_type = data_type
        self.jpl_storage, self.num_base_checkpoints, self.num_adapt_checkpoints = self.get_jpl_information()
        self.random_active_learning = RandomActiveLearning()
        self.confidence_active_learning = LeastConfidenceActiveLearning()
        
        self.use_gpu = use_gpu
        self.testing = testing

        self.batch_size = batch_size
        self.num_workers = num_workers
        
    def get_class(self):
        """
        TODO: This needs to not be mnist-specific
        :return: map from DataLoader class labels to SCADS node IDs
        """
        classes = {
            0: '/c/en/zero/n/wn/quantity',
            1: '/c/en/one/n/wn/quantity',
            2: '/c/en/two/n/wn/quantity',
            3: '/c/en/three/n/wn/quantity',
            4: '/c/en/four/n/wn/quantity',
            5: '/c/en/five/n/wn/quantity',
            6: '/c/en/six/n/wn/quantity',
            7: '/c/en/seven/n/wn/quantity',
            8: '/c/en/eight/n/wn/quantity',
            9: '/c/en/nine/n/wn/quantity',
        }
        return classes


    def get_jpl_information(self):
        jpl_task_names = self.api.get_available_tasks('image_classification')
        #### Elaheh: (need change in eval) choose image classification task you would like. Now there are four tasks
        image_classification_task = jpl_task_names[3]
        jpl_task_name = image_classification_task
        self.api.create_session(jpl_task_name)
        jpl_task_metadata = self.api.get_task_metadata(jpl_task_name)

        if self.api.data_type == 'full':
            num_base_checkpoints = len(jpl_task_metadata['base_label_budget_full'])
            num_adapt_checkpoints = len(jpl_task_metadata['adaptation_label_budget_full'])
        elif self.api.data_type == 'sample':
            num_base_checkpoints = len(jpl_task_metadata['base_label_budget_sample'])
            num_adapt_checkpoints = len(jpl_task_metadata['adaptation_label_budget_sample'])


        jpl_storage = JPLStorage(jpl_task_name, jpl_task_metadata)

        return jpl_storage, num_base_checkpoints, num_adapt_checkpoints

    def update_jpl_information(self):
        session_status = self.api.get_session_status()

        current_dataset = session_status['current_dataset']
        self.jpl_storage.classes = current_dataset['classes']
        self.jpl_storage.number_of_channels = current_dataset['number_of_channels']
        
        self.jpl_storage.phase = session_status['pair_stage']
        if session_status['pair_stage'] == 'adaptation':
            self.jpl_storage.set_image_path(self.adapt_dataset_dir, self.api.data_type)
        elif session_status['pair_stage'] == 'base':
            self.jpl_storage.set_image_path(self.base_dataset_dir, self.api.data_type)

    def run_checkpoints(self):
        self.run_checkpoints_base()
        self.run_checkpoints_adapt()

    def run_checkpoints_base(self):
        self.update_jpl_information()
        for i in range(self.num_base_checkpoints):
            self.run_one_checkpoint("Base", i)
    
    def run_checkpoints_adapt(self):
        self.update_jpl_information()
        for i in range(self.num_base_checkpoints):
            self.run_one_checkpoint("Adapt", i)

    
    def run_one_checkpoint(self, phase, checkpoint_num):
        log.info('------------------------------------------------------------')
        log.info('--------------------{} Checkpoint: {}'.format(phase, checkpoint_num)+'---------------------')
        log.info('------------------------------------------------------------')

        available_budget = self.get_available_budget()
        if checkpoint_num == 0:
            self.jpl_storage.labeled_images = self.api.get_initial_seed_labels()
        elif checkpoint_num == 1:
            self.jpl_storage.add_labeled_images(self.api.get_secondary_seed_labels())


        unlabeled_image_names = self.jpl_storage.get_unlabeled_image_names()
        log.info('number of unlabeled data: {}'.format(len(unlabeled_image_names)))
        if checkpoint_num == 2: #Elaheh: maybe we could get rid of random active learning?!
            candidates = self.random_active_learning.find_candidates(available_budget, unlabeled_image_names)
            self.request_labels(candidates)

        elif checkpoint_num > 2:
            candidates = self.confidence_active_learning.find_candidates(available_budget, unlabeled_image_names)
            self.request_labels(candidates)

        labeled_dataset, val_dataset = self.jpl_storage.get_labeled_dataset()
        unlabeled_dataset = self.jpl_storage.get_unlabeled_dataset()
        classes = self.get_class()
        task = Task(self.jpl_storage.name,
                    classes,
                    (28, 28),
                    labeled_dataset,
                    unlabeled_dataset,
                    val_dataset,
                    None)
        task.set_initial_model(MnistResNet())
        controller = Controller(task, self.batch_size, self.num_workers, self.use_gpu)
        end_model = controller.train_end_model()

        evaluation_dataset = self.jpl_storage.get_evaluation_dataset()
        evaluation_data_loader = torch.utils.data.DataLoader(evaluation_dataset,
                                                             batch_size=self.batch_size,
                                                             shuffle=False,
                                                             num_workers=self.num_workers)
        predictions, _ = end_model.predict(evaluation_data_loader, self.use_gpu)
        predictions_dict = {'id': self.jpl_storage.get_evaluation_image_names(), 'class': predictions}

        self.submit_predictions(predictions_dict)

        unlabeled_data_loader = torch.utils.data.DataLoader(unlabeled_dataset,
                                                            batch_size=self.batch_size,
                                                            shuffle=False,
                                                            num_workers=self.num_workers)
        _, confidences = end_model.predict(unlabeled_data_loader, self.use_gpu)
        candidates = np.argsort(confidences)
        self.confidence_active_learning.set_candidates(candidates)

    def get_available_budget(self):
        session_status = self.api.get_session_status()
        available_budget = session_status['budget_left_until_checkpoint']

        if self.testing:
            available_budget = available_budget // 10
        return available_budget

    def request_labels(self, examples):
        query = {'example_ids': examples}
        labeled_images = self.api.request_label(query)

        self.jpl_storage.add_labeled_images(labeled_images)
        log.info("New labeled images: %s", len(labeled_images))
        log.info("Total labeled images: %s", len(self.jpl_storage.labeled_images))

    def submit_predictions(self, predictions):
        print('**** session status after submit prediction  ****')
        submit_status = self.api.submit_prediction(predictions)
        print(submit_status)
        # session_status = self.api.get_session_status()
        session_status = submit_status
        if 'checkpoint_scores' in session_status:
            log.info("Checkpoint scores: %s", session_status['checkpoint_scores'])
        if 'pair_stage' in session_status:
            log.info("Phase: %s", session_status['pair_stage'])


def main():
    parser = argparse.ArgumentParser(description="Run JPL task")
    parser.add_argument("--dataset_dir", dest="dataset_dir",
                        type=str,
                        default="",
                        help="The directory to the dataset (in the case base and adapt datasets are the same)")
    parser.add_argument("--base_dataset_dir", dest="base_dataset_dir",
                        type=str,
                        default="/data/bats/datasets/lwll/lwll_datasets/development/mnist",
                        help="The directory to the base dataset")
    parser.add_argument("--adapt_dataset_dir", dest="adapt_dataset_dir",
                        type=str,
                        default="/data/bats/datasets/lwll/lwll_datasets/development/mnist",
                        help="The directory to the adapt dataset")
    args = parser.parse_args()
    
    if args.dataset_dir != "":
        base_dataset_dir = args.dataset_dir
        adapt_dataset_dir = args.dataset_dir
    else:
        base_dataset_dir = args.base_dataset_dir
        adapt_dataset_dir = args.adapt_dataset_dir
     
    logger = logging.getLogger()
    logger.level = logging.INFO
    stream_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    runner = JPLRunner(base_dataset_dir, adapt_dataset_dir, use_gpu=False, testing=False)
    runner.run_checkpoints()


if __name__ == "__main__":
    main()
