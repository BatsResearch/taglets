import os
import sys
import time
import logging
import argparse
import requests
import linecache
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import torchvision.models as models
import torchvision.transforms as transforms

import logger
from ..task import Task
from ..data import CustomDataset
from ..controller import Controller
from .utils import labels_to_concept_ids
from ..active import RandomActiveLearning, LeastConfidenceActiveLearning


gpu_list = os.getenv("LWLL_TA1_GPUS")
if gpu_list is not None and gpu_list != "all":
    gpu_list = [x for x in gpu_list.split(" ")]
    os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(gpu_list)

log = logging.getLogger(__name__)


class JPL:
    """
    A class to interact with JPL-like APIs.
    """
    def __init__(self, api_url, team_secret, gov_team_secret, dataset_type):
        """
        Create a new JPL object.
        """

        self.team_secret = team_secret
        self.gov_team_secret = gov_team_secret
        self.url = api_url 
        self.session_token = ''
        self.data_type = dataset_type

    def get_available_tasks(self, problem_type):
        """
        Get all available tasks.
        :return: A list of tasks (problems)
        """
        headers = {'user_secret': self.team_secret,
                   'govteam_secret': self.gov_team_secret
                   }
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
        headers = {'user_secret': self.team_secret,
                   'govteam_secret': self.gov_team_secret}
        
        r = requests.get(self.url + "/task_metadata/" + task_name, headers=headers)
        return r.json()['task_metadata']

    def create_session(self, task_name):
        """
        Create a new session.
        :param task_name: The name of the task (problem
        :return: None
        """
        headers = {'user_secret': self.team_secret,
                   'govteam_secret': self.gov_team_secret}

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
        headers = {'user_secret': self.team_secret,
                   'govteam_secret': self.gov_team_secret,
                   'session_token': self.session_token}
        r = requests.get(self.url + "/session_status", headers=headers)
        if 'Session_Status' in r.json():
            return r.json()['Session_Status']
        else:
            return {}

    def get_initial_seed_labels(self, video=False):
        """
        Get seed labels.
        :return: A list of lists with name and label e.g., ['2', '1.png'], ['7', '2.png'], etc.
        """

        log.info('Request seed labels.')
        headers = {'user_secret': self.team_secret,
                   'govteam_secret': self.gov_team_secret,
                   'session_token': self.session_token}
        r = requests.get(self.url + "/seed_labels", headers=headers)
        labels = r.json()['Labels']
        log.debug(f"NUM OF NEW RAW RETRIEVED LABELS: {len(labels)}")

        if video:
            seed_labels = []
            dictionary_clips = {}
            for clip in labels:
                action_frames = [str(i)+'.jpg' for i in range(clip['start_frame'], clip['end_frame'])]
                dictionary_clips[clip["id"]] = action_frames
                seed_labels.append([clip["class"], clip["id"]])
            return seed_labels, dictionary_clips

        else:
            seed_labels = []
            for image in labels:
                seed_labels.append([image["class"], image["id"]])
            return seed_labels, None

    def deactivate_session(self, deactivate_session):

        headers_active_session = {'user_secret': self.team_secret,
                                  'govteam_secret': self.gov_team_secret,
                                  'session_token': self.session_token}

        r = requests.post(self.url + "/deactivate_session",
                          json={'session_token': deactivate_session},
                          headers=headers_active_session)
        r.json()

    def request_label(self, query, video=False):
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
        headers = {'user_secret': self.team_secret,
                   'govteam_secret': self.gov_team_secret,
                   'session_token': self.session_token}
        log.debug(f"Query for new labels: {type(query['example_ids'][0])}")
        r = requests.post(self.url + "/query_labels", json=query, headers=headers)
        labels = r.json()['Labels']
        log.debug(f"NUM OF NEW RAW RETRIEVED LABELS: {len(labels)}")

        if video:
            labels_list = []
            dictionary_clips = {}
            for clip in labels:
                action_frames = [str(i)+'.jpg' for i in range(clip['start_frame'], clip['end_frame'])]
                dictionary_clips[clip["id"]] = action_frames
                labels_list.append([clip["class"], clip["id"]])
            return labels_list, dictionary_clips
        else:
            labels_list = []
            for image in labels:
                labels_list.append([image["class"], image["id"]])
            return labels_list, None

    def submit_prediction(self, predictions):
        """
        Submit predictions on test images.

        :param predictions: A dictionary containing test image names and corresponding labels.
        For example:
        predictions = {'id': ['6831.png', '1186.png', '8149.png', '4773.png', '3752.png'],
                       'label': ['9', '6', '9', '2', '10']}
        :return: The session status after submitting prediction
        """

        headers = {'user_secret': self.team_secret,
                   'govteam_secret': self.gov_team_secret,
                   'session_token': self.session_token}
        r = requests.post(self.url + "/submit_predictions", json={'predictions': predictions}, headers=headers)
        return r.json()

    def deactivate_all_sessions(self):

        headers_session = {'user_secret': self.team_secret,
                           'govteam_secret': self.gov_team_secret}
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
        if self.problem_type == 'video_classification':
            self.video = True
            log.info("We are running the video classification task")
        else: 
            self.video = False
            log.info("We are running the image classification task")
        self.task_id = metadata['task_id']
        self.classes = []
        self.evaluation_meta_path = "path to meta data for test videos"
        self.evaluation_image_path = "path to test images"
        self.unlabeled_image_path = "path to unlabeled images"
        self.unlabeled_meta_path = "path to meta data for train videos"
        self.labeled_images = []  # A list of tuples with name and label e.g., ['1.png', '2'], ['2.png', '7'], etc.
        self.dictionary_clips = None
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
        
    def set_image_path(self, dataset_dir, data_type, video=False):
        """
        Set self.evaluation_image_path and self.unlabeled_image_path with the given dataset_dir
        :param dataset_dir: the directory to the dataset
        :param data_type: 'sample' or 'full'
        :return:
        """
        
        self.unlabeled_image_path = os.path.join(dataset_dir,
                                                 os.path.basename(dataset_dir) + "_" + data_type,
                                                 "train")
        if video:
            self.evaluation_image_path = os.path.join(dataset_dir,
                                                      os.path.basename(dataset_dir) + "_" + data_type,
                                                      "test")
            self.evaluation_meta_path = os.path.join(dataset_dir,
                                                     "labels" + "_" + data_type,
                                                     "meta_test.feather")
            self.unlabeled_meta_path = os.path.join(dataset_dir,
                                                 "labels" + "_" + data_type,
                                                 "meta_train.feather")

        else:
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
        # Remember to check it for video and eval
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

        image_labels, image_names = list(zip(*self.labeled_images))

        return image_names, image_labels

    def get_unlabeled_image_names(self, dictionary_clips=None, video=False):
        """return list of name of unlabeled images"""
        
        if video: 
            labeled_clip_names = list(dictionary_clips.keys())
            train_meta = pd.read_feather(self.unlabeled_meta_path)
            unlabeled_clip_names = []
            for clip in train_meta.iterrows(): 
                row = clip[1]['id']
                if row not in labeled_clip_names:
                    unlabeled_clip_names.append(int(row))
            return unlabeled_clip_names

        else:
            labeled_image_names = [img_name for label, img_name in self.labeled_images]
            
            unlabeled_image_names = []
            for img in os.listdir(self.unlabeled_image_path):
                if img not in labeled_image_names:
                    unlabeled_image_names.append(img)
            return unlabeled_image_names

    def get_evaluation_image_names(self, video=False):

        if video:
            test_meta = pd.read_feather(self.evaluation_meta_path)
            evaluation_image_names = test_meta['id'].tolist()
        else:
            evaluation_image_names = []
            for img in os.listdir(self.evaluation_image_path):
                evaluation_image_names.append(img)
        return evaluation_image_names

    def get_labeled_dataset(self, checkpoint_num, dictionary_clips, video=False):
        """
        Get training, validation, and testing data loaders from labeled data.
        :return: Training, validation, and testing data loaders
        """
        
        image_names, image_labels = self.get_labeled_images_list()
        image_paths = [os.path.join(self.unlabeled_image_path, str(image_name)) for image_name in image_names]
        
        if video:
            paths_dictionary_clips = {}
            for clip, frames in dictionary_clips.items():
                paths_dictionary_clips[clip] = [os.path.join(self.unlabeled_image_path, str(f)) for f in frames]
            dictionary_clips = paths_dictionary_clips
        else:
            dictionary_clips = None
        
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
                                          transform=self.transform_image(train=True),
                                          video=self.video,
                                          clips_dictionary=dictionary_clips)
            val_dataset = CustomDataset(image_paths[val_idx],
                                        labels=image_labels[val_idx],
                                        label_map=self.label_map,
                                        transform=self.transform_image(train=False),
                                        video=self.video,
                                        clips_dictionary=dictionary_clips)
        else:
            train_dataset = CustomDataset(image_paths,
                                          labels=image_labels,
                                          label_map=self.label_map,
                                          transform=self.transform_image(train=True),
                                          video=self.video,
                                          clips_dictionary=dictionary_clips)
            val_dataset = None

        return train_dataset, val_dataset

    def get_unlabeled_dataset(self, train=True, video=False):
        """
        Get a data loader from unlabeled data.
        :return: A data loader containing unlabeled data
        """
        
        transform = self.transform_image(train=train)
        
        if video:
            image_paths = []
            dictionary_clips = {}
            train_meta = pd.read_feather(self.unlabeled_meta_path)
            for clip in train_meta.iterrows():
                row = clip[1]
                action_frames = [os.path.join(self.unlabeled_image_path, str(i)+'.jpg')
                                 for i in range(row['start_frame'], row['end_frame'])]
                dictionary_clips[row["id"]] = action_frames
                image_paths.append(os.path.join(self.unlabeled_image_path, str(row["id"])))
        else:
            image_names = self.get_unlabeled_image_names(None, video)
            
            image_paths = [os.path.join(self.unlabeled_image_path, image_name) for image_name in image_names]
            dictionary_clips = None
        
        if len(image_paths) == 0:
            return None
        else:
            return CustomDataset(image_paths,
                                transform=transform, 
                                video=video, 
                                clips_dictionary=dictionary_clips)

    def get_evaluation_dataset(self, video=False):
        """
        Get a data loader from evaluation/test data.
        :return: A data loader containing unlabeled data
        """
        transform = self.transform_image(train=False)
        
        if video:
            image_paths = []
            dictionary_clips = {}
            test_meta = pd.read_feather(self.evaluation_meta_path)
            for clip in test_meta.iterrows():
                row = clip[1]
                action_frames = [os.path.join(self.evaluation_image_path, str(i)+'.jpg')
                                 for i in range(row['start_frame'], row['end_frame'])]
                dictionary_clips[row["id"]] = action_frames
                image_paths.append(os.path.join(self.evaluation_image_path, str(row["id"])))

        else:
            evaluation_image_names = []
            for img in os.listdir(self.evaluation_image_path):
                evaluation_image_names.append(img)
            image_paths = [os.path.join(self.evaluation_image_path, image_name)
                           for image_name in evaluation_image_names]
            dictionary_clips = None
        
        return CustomDataset(image_paths,
                             transform=transform,
                             video=self.video,
                             clips_dictionary=dictionary_clips)


class JPLRunner:
    def __init__(self, dataset_type, problem_type, dataset_dir, api_url, problem_task, team_secret, gov_team_secret,
                 data_paths, simple_run=False, testing=False):

        self.dataset_dir = dataset_dir
        self.problem_type = problem_type
        if self.problem_type == 'video_classification':
            self.video = True
        else: 
            self.video = False
        self.api = JPL(api_url, team_secret, gov_team_secret, dataset_type)
        self.api.data_type = dataset_type
        self.problem_task = problem_task
        self.data_paths = data_paths
        self.jpl_storage, self.num_base_checkpoints, self.num_adapt_checkpoints = self.get_jpl_information()
        self.random_active_learning = RandomActiveLearning()
        self.confidence_active_learning = LeastConfidenceActiveLearning()

        self.initial_model = models.resnet50(pretrained=True)
        self.initial_model.fc = torch.nn.Identity()

        self.testing = testing
        self.simple_run = simple_run

    def get_jpl_information(self):
        # Elaheh: (need change in eval) choose image classification task you would like. Now there are four tasks
        image_classification_task = self.problem_task
        jpl_task_name = image_classification_task
        self.api.create_session(jpl_task_name)
        jpl_task_metadata = self.api.get_task_metadata(jpl_task_name)
        log.info(f"Task metadata: {jpl_task_metadata}")

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
        self.jpl_storage.set_image_path(phase_dataset_dir, self.api.data_type, self.video)

    def run_checkpoints(self):
        try:
            self.run_checkpoints_base()
            self.run_checkpoints_adapt()
        except Exception as ex:
            #exc_type, exc_obj, tb = sys.exc_info()
            #f = tb.tb_frame
            #lineno = tb.tb_lineno
            #filename = f.f_code.co_filename
            #linecache.checkcache(filename)
            #line = linecache.getline(filename, lineno, f.f_globals)
            self.api.deactivate_session(self.api.session_token)

            logging.exception('EXCEPTION has occured during joint training:')
            #logging.info('exception has occured during joint training:')
            #logging.info(ex)
            #logging.info('EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj))

    def run_checkpoints_base(self):
        log.info("Enter checkpoint")
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
            self.jpl_storage.labeled_images, self.jpl_storage.dictionary_clips = \
                self.api.get_initial_seed_labels(self.video)
            log.info(f'Get initial seeds at {checkpoint_num} checkpoint')
        elif 1 <= checkpoint_num <= 3:
            new_labeled_images, new_dictionary_clips = self.api.get_initial_seed_labels(self.video)
            self.jpl_storage.add_labeled_images(new_labeled_images)
            if self.jpl_storage.dictionary_clips != None:
                self.jpl_storage.dictionary_clips.update(new_dictionary_clips)
            log.info(f'Get seeds at {checkpoint_num} checkpoints')
        
        # Get sets of unlabeled samples
        unlabeled_image_names = self.jpl_storage.get_unlabeled_image_names(self.jpl_storage.dictionary_clips,
                                                                           self.video)
        log.debug('Number of unlabeled data: {}'.format(len(unlabeled_image_names)))
        
        if checkpoint_num >= 4:
            """ For the last evaluation we used to start asking for custom labels after the first 2 checkpoints.
            Moreover we adopted randomActive learning for the first query. Do we want it?

            candidates = self.random_active_learning.find_candidates(available_budget, unlabeled_image_names)
            self.request_labels(candidates)
            """

            """ For the hackathon we only use RandomActiveLearning - bring it back

            candidates = self.confidence_active_learning.find_candidates(available_budget, unlabeled_image_names)
            """
            # Pick candidates from the list
            """ To consider: we directly query for all the available budget, we have the option 
            of gradually ask new labels untile we exhaust the budget.
            """
            candidates = self.random_active_learning.find_candidates(available_budget, unlabeled_image_names)
            log.debug(f"Candidates to query[0]: {candidates[0]}")
            self.request_labels(candidates, self.video)

        labeled_dataset, val_dataset = self.jpl_storage.get_labeled_dataset(checkpoint_num, self.jpl_storage.dictionary_clips, self.video)
        unlabeled_train_dataset = self.jpl_storage.get_unlabeled_dataset(True, self.video)
        unlabeled_test_dataset = self.jpl_storage.get_unlabeled_dataset(False, self.video)
        # sys.exit()
        task = Task(self.jpl_storage.name,
                    labels_to_concept_ids(self.jpl_storage.classes),
                    (224, 224), 
                    labeled_dataset,
                    unlabeled_train_dataset,
                    val_dataset,
                    self.jpl_storage.whitelist,
                    self.data_paths[0],
                    self.data_paths[1],
                    unlabeled_test_data=unlabeled_test_dataset,
                    video_classification=self.video)
        task.set_initial_model(self.initial_model)
        controller = Controller(task, self.simple_run)
        
        # sys.exit()

        end_model = controller.train_end_model()

        evaluation_dataset = self.jpl_storage.get_evaluation_dataset(self.video)
        outputs = end_model.predict(evaluation_dataset)
        predictions = np.argmax(outputs, 1)
        prediction_names = []
        for p in predictions:
            prediction_names.append([k for k, v in self.jpl_storage.label_map.items() if v == p][0])

        predictions_dict = {'id': self.jpl_storage.get_evaluation_image_names(self.video), 'class': prediction_names}

        self.submit_predictions(predictions_dict)

        if unlabeled_test_dataset is not None:
            outputs = end_model.predict(unlabeled_test_dataset)
            confidences = np.max(outputs, 1)
            candidates = np.argsort(confidences)
            self.confidence_active_learning.set_candidates(candidates)

        # update initial model
        if checkpoint_num == 7:
            self.initial_model = end_model.model
            self.initial_model.fc = torch.nn.Identity()

        log.info('{} Checkpoint: {} Elapsed Time =  {}'.format(phase,
                                                               checkpoint_num,
                                                               time.strftime("%H:%M:%S",
                                                                             time.gmtime(time.time()-start_time))))

    def get_available_budget(self):
        session_status = self.api.get_session_status()
        available_budget = session_status['budget_left_until_checkpoint']

        if self.testing:
            available_budget = available_budget // 10
        return available_budget

    def request_labels(self, examples, video=False):
        query = {'example_ids': examples}
        labeled_images, labeled_dictionary_clips = self.api.request_label(query, video)

        self.jpl_storage.add_labeled_images(labeled_images)
        if self.jpl_storage.dictionary_clips != None:
            self.jpl_storage.dictionary_clips.update(labeled_dictionary_clips)
        log.info("New labeled images: %s", len(labeled_images))
        log.info("Total labeled images: %s", len(self.jpl_storage.labeled_images))

    def submit_predictions(self, predictions):
        log.info('**** session status after submit prediction  ****')
        submit_status = self.api.submit_prediction(predictions)
        log.info(submit_status)
        # session_status = self.api.get_session_status()
        session_status = submit_status
        if 'checkpoint_scores' in session_status:
            log.info("Checkpoint scores: %s", session_status['checkpoint_scores'])
        if 'pair_stage' in session_status:
            log.info("Phase: %s", session_status['pair_stage'])


def workflow(dataset_type, problem_type, dataset_dir, api_url, problem_task, team_secret, gov_team_secret, data_paths,
             simple_run):
    if problem_task == 'all':
        log.info('Execute all tasks')
        print(log.info('Execute all tasks'))
        jpl = JPL(api_url, team_secret, gov_team_secret, dataset_type)
        problem_task_list = jpl.get_available_tasks(problem_type)
        for task in problem_task_list:
            runner = JPLRunner(dataset_type, problem_type, dataset_dir, api_url, task, team_secret, gov_team_secret,
                               data_paths, simple_run, testing=False)
            runner.run_checkpoints()
    else:
        log.info("Execute a single task")
        runner = JPLRunner(dataset_type, problem_type, dataset_dir, api_url, problem_task, team_secret, gov_team_secret,
                           data_paths, simple_run, testing=False)
        runner.run_checkpoints()


def setup_production():
    """
    This function returns the variables needed to launch the system in production.
    """

    dataset_type = os.environ.get('LWLL_TA1_DATASET_TYPE')
    problem_type = os.environ.get('LWLL_TA1_PROB_TYPE')
    dataset_dir = os.environ.get('LWLL_TA1_DATA_PATH')
    api_url = os.environ.get('LWLL_TA1_API_ENDPOINT')
    problem_task = os.environ.get('LWLL_TA1_PROB_TASK')
    gpu_list = os.environ.get('LWLL_TA1_GPUS')
    run_time = os.environ.get('LWLL_TA1_HOURS')
    team_secret = os.environ.get('LWLL_TA1_TEAM_SECRET')
    gov_team_secret = os.environ.get('LWLL_TA1_GOVTEAM_SECRET')
    data_paths = ('/tmp/predefined/scads.fall2020.sqlite3',
                  '/tmp/predefined/embeddings/numberbatch-en19.08.txt.gz')

    # check gpus are all
    if gpu_list != 'all':
        raise Exception(f'all gpus are required')
    
    return dataset_type, problem_type, dataset_dir, api_url, problem_task, team_secret, gov_team_secret, data_paths


def setup_development():
    """
    This function returns the variables needed to launch the system in development.
    """

    # not sure this is very elegant. Let me know :)
    import dev_config

    return (dev_config.dataset_type, dev_config.problem_type, dev_config.dataset_dir, dev_config.api_url,
            dev_config.problem_task, dev_config.team_secret, dev_config.gov_team_secret, dev_config.data_paths)


def main():
    
    parser = argparse.ArgumentParser(description="Run JPL task")
    parser.add_argument("--mode",
                        type=str,
                        default="prod",
                        help="The mode to execute the system. prod: system eval, dev: system development")
    parser.add_argument("--simple_version",
                        type=str,
                        default="false",
                        help="Option to choose whether exclude or not the real train")
    args = parser.parse_args()
    
    if args.mode == 'prod':
        variables = setup_production()
    else:
        variables = setup_development()
    
    dataset_type = variables[0]
    problem_type = variables[1]
    log.info(f"Problem type: {problem_type}")
    dataset_dir = variables[2]
    api_url = variables[3]
    problem_task = variables[4]
    team_secret = variables[5]
    gov_team_secret = variables[6]
    data_paths = variables[7]
    if args.simple_version == 'true':
        simple_run = True
    else: 
        simple_run = False

    valid_dataset_types = ['sample', 'full', 'all']
    if dataset_type not in valid_dataset_types:
        raise Exception(f'Invalid `dataset_type`, expected one of {valid_dataset_types}')

    # Check problem type is valid
    valid_problem_types = ['image_classification', 'object_detection', 'machine_translation', 'video_classification',
                           'all']
    if problem_type not in valid_problem_types:
        raise Exception(f'Invalid `problem_type`, expected one of {valid_problem_types}')

    # Check dataset directory exists
    if not Path(dataset_dir).exists():
        raise Exception('`dataset_dir` does not exist..')

    logger = logging.getLogger(__name__)
    logger.level = logging.DEBUG
    stream_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    workflow(dataset_type, problem_type, dataset_dir, api_url, problem_task, team_secret, gov_team_secret, data_paths,
             simple_run)
    

if __name__ == "__main__":
    main()
