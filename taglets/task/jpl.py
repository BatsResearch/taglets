import os
gpu_list = os.getenv("LWLL_TA1_GPUS")
if gpu_list is not None and gpu_list != "all":
    gpu_list = [x for x in gpu_list.split(" ")]
    os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(gpu_list)
import argparse
import logging
import sys
import time
import requests
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils import data
from ..data import CustomDataset
from ..active import RandomActiveLearning, LeastConfidenceActiveLearning
from ..task import Task
from ..controller import Controller
from ..scads import Scads
from .utils import labels_to_concept_ids
import linecache
import pickle


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
        self.data_type = 'full'   # Sample or full

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
        :return: A list of lists with name and label e.g., ['2', '1.png'], ['7', '2.png'], etc.
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

    def deactivate_session(self, deactivate_session):

        headers_active_session = {'user_secret': self.secret, 'session_token': self.session_token}

        r = requests.post(self.url + "/deactivate_session",
                          json={'session_token': deactivate_session},
                          headers=headers_active_session)
        r.json()

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

        :return: A list of lists containing labels and names.
        For example:
         [['7','56392.png'], ['8','3211.png'], ['4','19952.png']]
        """
        headers = {'user_secret': self.secret, 'session_token': self.session_token}
        r = requests.post(self.url + "/query_labels", json=query, headers=headers)
        labels_dic = r.json()['Labels']
        labels_list = [(d['class'], d['id']) for d in labels_dic]
        return labels_list

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

    def deactivate_all_sessions(self):

        headers_session = {'user_secret': self.secret}
        r = requests.get(self.url + "/list_active_sessions", headers=headers_session)
        active_sessions = r.json()['active_sessions']
        for session_token in active_sessions:
            self.deactivate_session(session_token)

    
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
        self.whitelist = metadata['whitelist']

        self.label_map = {}
    
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

    def transform_image(self, train=True):
        """
        Get the transform to be used on an image.
        :return: A transform
        """
        data_mean = [0.485, 0.456, 0.406]
        data_std = [0.229, 0.224, 0.225]
    
        if train:
            return transforms.Compose([
                transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=data_mean, std=data_std)
            ])
        else:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=data_mean, std=data_std)
            ])

    def get_labeled_images_list(self):
        """get list of image names and labels"""
        # Elaheh: changed the following line
        image_labels = [item[0] for item in self.labeled_images]
        image_names = [item[1] for item in self.labeled_images]

        # image_labels = sorted(image_labels)

        return image_names, image_labels

    def get_unlabeled_image_names(self):
        """return list of name of unlabeled images"""
        labeled_image_names = [img_name for label, img_name in self.labeled_images]
        unlabeled_image_names = []
        for img in os.listdir(self.unlabeled_image_path):
            if img not in labeled_image_names:
                unlabeled_image_names.append(img)
        return unlabeled_image_names
    
    def get_train_image_names(self):
        """return list of name of unlabeled images"""
        train_image_names = []
        for img in os.listdir(self.unlabeled_image_path):
            train_image_names.append(img)
        return train_image_names

    def get_evaluation_image_names(self):
        evaluation_image_names = []
        for img in os.listdir(self.evaluation_image_path):
            evaluation_image_names.append(img)
        return evaluation_image_names

    def get_labeled_dataset(self, checkpoint_num):
        """
        Get training, validation, and testing data loaders from labeled data.
        :return: Training, validation, and testing data loaders
        """
        image_names, image_labels = self.get_labeled_images_list()
        image_paths = [os.path.join(self.unlabeled_image_path, image_name) for image_name in image_names]
        image_paths = np.asarray(image_paths)
        image_labels = np.asarray(image_labels)
        

        if checkpoint_num >= 2:
            # 80% for training, 20% for validation
            train_percent = 0.8
            num_data = len(image_paths)
            indices = list(range(num_data))
            train_split = int(np.floor(train_percent * num_data))
            np.random.shuffle(indices)
            train_idx = indices[:train_split]
            val_idx = indices[train_split:]

            train_dataset = CustomDataset(image_paths[train_idx],
                                          labels=image_labels[train_idx],
                                          label_map=self.label_map,
                                          transform=self.transform_image(train=True))
            val_dataset = CustomDataset(image_paths[val_idx],
                                        labels=image_labels[val_idx],
                                        label_map=self.label_map,
                                        transform=self.transform_image(train=False))
        else:
            train_dataset = CustomDataset(image_paths,
                                          labels=image_labels,
                                          label_map=self.label_map,
                                          transform=self.transform_image(train=True))
            val_dataset = None

        return train_dataset, val_dataset

    def get_unlabeled_dataset(self, train=True):
        """
        Get a data loader from unlabeled data.
        :return: A data loader containing unlabeled data
        """
        transform = self.transform_image(train=train)

        image_names = self.get_unlabeled_image_names()
        image_paths = [os.path.join(self.unlabeled_image_path, image_name) for image_name in image_names]
        if len(image_paths) == 0:
            return None
        else:
            return CustomDataset(image_paths,
                                 transform=transform)
        
    def get_train_dataset(self):
        """
        Get a data loader from unlabeled data.
        :return: A data loader containing unlabeled data
        """
        transform = self.transform_image(train=False)

        image_names = self.get_train_image_names()
        image_paths = [os.path.join(self.unlabeled_image_path, image_name) for image_name in image_names]
        return CustomDataset(image_paths,
                             transform=transform)

    def get_evaluation_dataset(self):
        """
        Get a data loader from evaluation/test data.
        :return: A data loader containing unlabeled data
        """
        transform = self.transform_image(train=False)

        evaluation_image_names = []
        for img in os.listdir(self.evaluation_image_path):
            evaluation_image_names.append(img)
        image_paths = [os.path.join(self.evaluation_image_path, image_name) for image_name in evaluation_image_names]
        return CustomDataset(image_paths,
                             transform=transform)


class JPLRunner:
    def __init__(self, dataset_dir, task_ix, testing=False, data_type='full'):
        self.dataset_dir = dataset_dir

        self.api = JPL()
        self.api.data_type = data_type
        self.task_ix = task_ix
        self.jpl_storage, self.num_base_checkpoints, self.num_adapt_checkpoints = self.get_jpl_information()
        self.random_active_learning = RandomActiveLearning()
        self.confidence_active_learning = LeastConfidenceActiveLearning()

        self.initial_model = models.resnet50(pretrained=True)
        self.initial_model.fc = torch.nn.Identity()

        self.testing = testing
        self.ta2 = {}

    def get_jpl_information(self):
        jpl_task_names = self.api.get_available_tasks('image_classification')
        # Elaheh: (need change in eval) choose image classification task you would like. Now there are four tasks
        image_classification_task = jpl_task_names[self.task_ix]
        jpl_task_name = image_classification_task
        self.api.create_session(jpl_task_name)
        jpl_task_metadata = self.api.get_task_metadata(jpl_task_name)
        print('jpl_task_metadata')
        print(jpl_task_metadata)

        if self.api.data_type == 'full':
            num_base_checkpoints = len(jpl_task_metadata['base_label_budget_full'])
            num_adapt_checkpoints = len(jpl_task_metadata['adaptation_label_budget_full'])
        else:
            num_base_checkpoints = len(jpl_task_metadata['base_label_budget_sample'])
            num_adapt_checkpoints = len(jpl_task_metadata['adaptation_label_budget_sample'])

        jpl_storage = JPLStorage(jpl_task_name, jpl_task_metadata)

        return jpl_storage, num_base_checkpoints, num_adapt_checkpoints

    def update_jpl_information(self):
        session_status = self.api.get_session_status()

        current_dataset = session_status['current_dataset']

        self.jpl_storage.classes = current_dataset['classes']
        self.jpl_storage.number_of_channels = current_dataset['number_of_channels']

        label_map = {}
        class_names = self.jpl_storage.classes
        for idx, item in enumerate(class_names):
            label_map[item] = idx

        self.jpl_storage.label_map = label_map

        self.jpl_storage.phase = session_status['pair_stage']
        phase_dataset_dir = os.path.join(self.dataset_dir, current_dataset['name'])
        self.jpl_storage.set_image_path(phase_dataset_dir, self.api.data_type)

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

        start_time = time.time()

        available_budget = self.get_available_budget()
        if checkpoint_num == 0:
            self.jpl_storage.labeled_images = self.api.get_initial_seed_labels()
        elif checkpoint_num == 1:
            self.jpl_storage.add_labeled_images(self.api.get_secondary_seed_labels())

        unlabeled_image_names = self.jpl_storage.get_unlabeled_image_names()
        log.info('number of unlabeled data: {}'.format(len(unlabeled_image_names)))
        if checkpoint_num >= 2:  # Elaheh: maybe we could get rid of random active learning?!
            candidates = self.random_active_learning.find_candidates(available_budget, unlabeled_image_names)
            self.request_labels(candidates)

        labeled_dataset, val_dataset = self.jpl_storage.get_labeled_dataset(checkpoint_num)
        train_dataset = self.jpl_storage.get_train_dataset()
        eval_dataset = self.jpl_storage.get_evaluation_dataset()
        task = Task(self.jpl_storage.name,
                    labels_to_concept_ids(self.jpl_storage.classes),
                    (224, 224),
                    labeled_dataset,
                    train_dataset,
                    val_dataset,
                    self.jpl_storage.whitelist,
                    'predefined/scads.fall2020.sqlite3',
                    'predefined/embeddings/numberbatch-en19.08.txt.gz',
                    unlabeled_test_data=eval_dataset)
        task.set_initial_model(self.initial_model)
        controller = Controller(task)
        vote_matrix1, vote_matrix2 = controller.train_end_model()

        predictions_dict = {'id': self.jpl_storage.get_evaluation_image_names(),
                            'class': [self.jpl_storage.classes[0]] * len(self.jpl_storage.get_evaluation_image_names())}
        self.submit_predictions(predictions_dict)

        log.info('{} Checkpoint: {} Elapsed Time =  {}'.format(phase,
                                                               checkpoint_num,
                                                               time.strftime("%H:%M:%S",
                                                                             time.gmtime(time.time()-start_time))))

        labeled_train_image_names, _ = self.jpl_storage.get_labeled_images_list()
        unlabeled_train_image_names, _ = self.jpl_storage.get_unlabeled_image_names()
        checkpoint_dict = {'train_images_names': self.jpl_storage.get_train_image_names(),
                           'train_images_votes': vote_matrix1,
                           'labeld_train_image_names': labeled_train_image_names,
                           'unlabeled_train_image_names': unlabeled_train_image_names,
                           'test_images_names': self.jpl_storage.get_evaluation_image_names(),
                           'test_images_votes': vote_matrix2}
        self.ta2[f'{phase} {checkpoint_num}'] = checkpoint_dict
        with open('ta2_votes.pkl', 'wb') as f:
            pickle.dump(self.ta2, f)

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
                        default="/lwll/development",
                        help="The directory to all development datasets")

    parser.add_argument("--scads_root_dir",
                        type=str,
                        default="/lwll/external",
                        help="The directory to all external datasets")

    parser.add_argument("--task_ix", type=int, default=0,
                        help="Index of image classification task; 0, 1, 2, etc.")

    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    scads_root_dir = args.scads_root_dir

    task_ix = args.task_ix
    logger = logging.getLogger()
    logger.level = logging.INFO
    stream_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    Scads.set_root_path(scads_root_dir)
    
    runner = JPLRunner(dataset_dir, task_ix, testing=False)
    runner.run_checkpoints()


if __name__ == "__main__":
    main()
