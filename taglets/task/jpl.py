import os
import sys
import datetime
import time
import logging
import argparse
import json
import requests
import pickle
from pathlib import Path
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

from accelerate import Accelerator
accelerator = Accelerator()
import torch
import numpy as np
import pandas as pd
import torchvision.models as models
import torchvision.transforms as transforms
import pytorchvideo.transforms as video_transform
import torchvision.transforms._transforms_video as transform_video

from ..task import Task
from ..data import CustomImageDataset, CustomVideoDataset, HandleExceptionCustomVideoDataset
from ..controller import Controller
from .utils import labels_to_concept_ids
from ..active import RandomActiveLearning, LeastConfidenceActiveLearning
from ..scads import Scads
from ..modules.videos.utils import PackPathway


gpu_list = os.getenv("LWLL_TA1_GPUS")
if gpu_list is not None and gpu_list != "all":
    gpu_list = [x for x in gpu_list.split(" ")]
    print(gpu_list)
    os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(gpu_list)

log = logging.getLogger(__name__)


DEFAULT_TIMEOUT = 10 # seconds


class TimeoutHTTPAdapter(HTTPAdapter):
    def __init__(self, *args, **kwargs):
        self.timeout = DEFAULT_TIMEOUT
        if "timeout" in kwargs:
            self.timeout = kwargs["timeout"]
            del kwargs["timeout"]
        super().__init__(*args, **kwargs)

    def send(self, request, **kwargs):
        timeout = kwargs.get("timeout")
        if timeout is None:
            kwargs["timeout"] = self.timeout
        return super().send(request, **kwargs)


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
        self.saved_api_response_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_api_response')
        
        retry_strategy = Retry(
            total=10,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = TimeoutHTTPAdapter(max_retries=retry_strategy)
        self.session = requests.Session()
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def get_available_tasks(self, problem_type):
        """
        Get all available tasks.
        :return: A list of tasks (problems)
        """
        headers = {'user_secret': self.team_secret,
                   'govteam_secret': self.gov_team_secret
                   }
        r = self.session.get(self.url + "/list_tasks", headers=headers)
        task_list = r.json()['tasks']

        subset_tasks = []
        for _task in task_list:
            r = self.session.get(self.url+"/task_metadata/"+_task, headers=headers)
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
        
        r = self.session.get(self.url + "/task_metadata/" + task_name, headers=headers)
        return r.json()['task_metadata']

    def create_session(self, task_name):
        """
        Create a new session.
        :param task_name: The name of the task (problem
        :return: None
        """
        headers = {'user_secret': self.team_secret,
                   'govteam_secret': self.gov_team_secret}
        session_json = {'session_name': 'testing', 'data_type': self.data_type, 'task_id': task_name}
        
        response = self.post_only_once("auth/create_session", headers, session_json)

        session_token = response['session_token']
        self.session_token = session_token

    def skip_checkpoint(self):
        """ Skip checkpoint.

        :return: session status after skipping checkpoint
        """
        headers = {'user_secret': self.team_secret,
                   'govteam_secret': self.gov_team_secret,
                   'session_token': self.session_token}
        self.get_only_once("skip_checkpoint", headers)
        # if 'Session_Status' in r.json():
        #     return r.json()['Session_Status']
        # else:
        #     return {}


    def get_session_status(self):
        """
        Get the session status.
        :return: The session status
        """
        headers = {'user_secret': self.team_secret,
                   'govteam_secret': self.gov_team_secret,
                   'session_token': self.session_token}
        r = self.session.get(self.url + "/session_status", headers=headers)
        if 'Session_Status' in r.json():
            return r.json()['Session_Status']
        else:
            return {}


#     def get_initial_seed_labels(self, video=False):
#         """
#         Get seed labels.
#         :return: A list of lists with name and label e.g., ['2', '1.png'], ['7', '2.png'], etc.
#         """

#         log.info('Request seed labels.')
#         headers = {'user_secret': self.team_secret,
#                    'govteam_secret': self.gov_team_secret,
#                    'session_token': self.session_token}
#         log.debug(f"HEADERS: {headers}")
#         response = self.get_only_once("seed_labels", headers)
#         labels = response['Labels']

#         if video:
#             seed_labels = []
#             dictionary_clips = {}
#             for clip in labels:
#                 action_frames = [str(clip['video_id']) + '/' + str(i)+'.jpg' for i in range(clip['start_frame'], clip['end_frame'] + 1)]
#                 dictionary_clips[clip["id"]] = action_frames
#                 seed_labels.append([clip["class"], clip["id"]])
#             return seed_labels, dictionary_clips

#         else:
#             seed_labels = []
#             for image in labels:
#                 seed_labels.append([image["class"], image["id"]])
#             return seed_labels, None


    def deactivate_session(self, deactivate_session):
        if accelerator.is_local_main_process:
            headers_active_session = {'user_secret': self.team_secret,
                                      'govteam_secret': self.gov_team_secret,
                                      'session_token': self.session_token}
    
            r = self.session.post(self.url + "/deactivate_session",
                              json={'session_token': deactivate_session},
                              headers=headers_active_session)

    def request_label(self, query=None, video=False):
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
        log.info(f'Headers: {headers}')
        if query is None:
            log.info('Requesting seed labels...')
            response = self.get_only_once("seed_labels", headers)
            labels = response['Labels']
        else:
            log.info('Requesting labels...')
            labels = []
            for i in range(0, len(query['example_ids']), 10000):
                batched_query = {'example_ids': query['example_ids'][i: i+10000]}
                log.info(f'Length of batched query {len(batched_query["example_ids"])}')
                log.info(f'Batched query ids {batched_query["example_ids"]}')
                response = self.post_only_once("query_labels", headers, batched_query)
                batched_labels = response['Labels']
                labels = labels + batched_labels
        log.info(f"Num of new raw retrieved labels: {len(labels)}")

        if video:
            labels_list = []
            dictionary_clips = {}
            for clip in labels:
                action_frames = [str(clip["video_id"]) + '/' + str(i)+'.jpg' for i in range(clip['start_frame'], clip['end_frame'] + 1)]
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
        predictions_json={'predictions': predictions}
        return self.post_only_once('submit_predictions', headers, predictions_json)
        
    def deactivate_all_sessions(self):
        headers_session = {'user_secret': self.team_secret,
                           'govteam_secret': self.gov_team_secret}
        r = self.session.get(self.url + "/list_active_sessions", headers=headers_session)
        active_sessions = r.json()['active_sessions']
        for session_token in active_sessions:
            self.deactivate_session(session_token)

    def post_only_once(self, command, headers, posting_json):
        if accelerator.is_local_main_process:
            r = self.session.post(self.url + "/" + command, json=posting_json, headers=headers)
            with open(os.path.join(self.saved_api_response_dir, command.replace("/", "_") + "_response.json"), "w") as f:
                json.dump(r.json(), f)
        accelerator.wait_for_everyone()
        with open(os.path.join(self.saved_api_response_dir, command.replace("/", "_") + "_response.json"), "r") as f:
            response = json.load(f)
        return response
    
    def get_only_once(self, command, headers):
        if accelerator.is_local_main_process:
            r = self.session.get(self.url + "/" + command, headers=headers)
            with open(os.path.join(self.saved_api_response_dir, command.replace("/", "_") + "_response.json"), "w") as f:
                json.dump(r.json(), f)
        accelerator.wait_for_everyone()
        with open(os.path.join(self.saved_api_response_dir, command.replace("/", "_") + "_response.json"), "r") as f:
            response = json.load(f)
        return response
    
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
        
    def set_image_path(self, dataset_dir, dataset_name, data_type, video=False):
        """
        Set self.evaluation_image_path and self.unlabeled_image_path with the given dataset_dir
        :param dataset_dir: the directory to the dataset
        :param data_type: 'sample' or 'full'
        :return:
        """
        
        self.unlabeled_image_path = os.path.join(dataset_dir,
                                                 dataset_name,
                                                 dataset_name + "_" + data_type,
                                                 "train")
        self.all_train_labels_path = os.path.join(dataset_dir,
                                            '..',
                                            'external',
                                            dataset_name,
                                            "labels" + "_" + data_type,
                                            "labels_train.feather")
        self.test_labels_path = os.path.join(dataset_dir,
                                                  '..',
                                                  'external',
                                                  dataset_name,
                                                  "labels" + "_" + data_type,
                                                  "labels_test.feather")
        if video:
            self.evaluation_image_path = os.path.join(dataset_dir,
                                                      dataset_name,
                                                      dataset_name + "_" + data_type,
                                                      "test")
            self.evaluation_meta_path = os.path.join(dataset_dir,
                                                     dataset_name,
                                                     "labels" + "_" + data_type,
                                                     "meta_test.feather")
            self.unlabeled_meta_path = os.path.join(dataset_dir,
                                                    dataset_name,
                                                    "labels" + "_" + data_type,
                                                    "meta_train.feather")

        else:
            self.evaluation_image_path = os.path.join(dataset_dir,
                                                      dataset_name,
                                                      dataset_name + "_" + data_type,
                                                      "test")
    
    def transform_image(self, train=True, video=False):
        """
        Get the transform to be used on an image.
        :return: A transform
        """
        data_mean = [0.485, 0.456, 0.406]
        data_std = [0.229, 0.224, 0.225]
        # Remember to check it for video and eval
        if video:
            return transforms.Compose([transforms.ToTensor()])
        
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

    def transformer_video(self):   
        """Trasformation valid for SlowFast"""
        side_size = 256
        mean = [0.45, 0.45, 0.45]
        std = [0.225, 0.225, 0.225]
        crop_size = 256
        num_frames = 32
        alpha = 4

        
        return  video_transform.ApplyTransformToKey(key="video",
                                    transform=transforms.Compose([
                                        video_transform.UniformTemporalSubsample(num_frames),
                                        #transforms.Lambda(lambda x: x/255.0),
                                        transform_video.NormalizeVideo(mean, std),# transform_video
                                        video_transform.ShortSideScale(size=side_size),
                                        transform_video.CenterCropVideo(crop_size),# transform_video
                                        PackPathway(alpha)
                                    ])
                                    )


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
                    unlabeled_clip_names.append(str(row))
            return unlabeled_clip_names

        else:
            labeled_image_names = [img_name for label, img_name in self.labeled_images]
            
            unlabeled_image_names = []
            for img in os.listdir(self.unlabeled_image_path):
                if img not in labeled_image_names:
                    unlabeled_image_names.append(str(img))
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

        if checkpoint_num >= 4:
            # 80% for training, 20% for validation
            train_percent = 0.8
            num_data = len(image_paths)
            indices = list(range(num_data))
            train_split = int(np.floor(train_percent * num_data))
            np.random.shuffle(indices)
            train_idx = indices[:train_split]
            val_idx = indices[train_split:]
        else:
            train_idx = list(range(len(image_paths)))
            val_idx = []
        
        if self.video:
            #log.info(f"Disctionary clip: {dictionary_clips}")
            train_dataset = HandleExceptionCustomVideoDataset(image_paths[train_idx],
                                               labels=image_labels[train_idx],
                                               label_map=self.label_map,
                                               transform_img=self.transform_image(video=self.video),
                                               transform_vid=self.transformer_video(),
                                               clips_dictionary=dictionary_clips)
            if len(val_idx) != 0:
                val_dataset = HandleExceptionCustomVideoDataset(image_paths[val_idx],
                                                 labels=image_labels[val_idx],
                                                 label_map=self.label_map,
                                                 transform_img=self.transform_image(video=self.video),
                                                 transform_vid=self.transformer_video(),
                                                 clips_dictionary=dictionary_clips)
            else:
                val_dataset = None
        else:
            train_dataset = CustomImageDataset(image_paths[train_idx],
                                               labels=image_labels[train_idx],
                                               label_map=self.label_map,
                                               transform=self.transform_image(train=True))
            if len(val_idx) != 0:
                val_dataset = CustomImageDataset(image_paths[val_idx],
                                                 labels=image_labels[val_idx],
                                                 label_map=self.label_map,
                                                 transform=self.transform_image(train=False))
            else:
                val_dataset = None

        return train_dataset, val_dataset

    def get_unlabeled_dataset(self, train=True, video=False, dict_clips=None):
        """
        Get a data loader from unlabeled data.
        :return: A data loader containing unlabeled data
        """
        
        transform = self.transform_image(train=train)
        
        if video:
            clip_names = self.get_unlabeled_image_names(dict_clips, video)
            image_paths = []
            dictionary_clips = {}
            train_meta = pd.read_feather(self.unlabeled_meta_path)
            for clip in train_meta.iterrows():
                row = clip[1]
                if row["id"] in clip_names:
                    action_frames = [os.path.join(self.unlabeled_image_path, str(row['video_id'])) + '/' + str(i)+'.jpg'
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
            if self.video:
                return HandleExceptionCustomVideoDataset(image_paths,
                                          transform_img=self.transform_image(train=train, video=self.video),
                                          transform_vid=self.transformer_video(),
                                          clips_dictionary=dictionary_clips)
            else:
                return HandleExceptionCustomVideoDataset(image_paths,
                                          transform=transform)

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
                action_frames = [os.path.join(self.evaluation_image_path, str(row['video_id'])) + '/' + str(i)+'.jpg'
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
        
        if self.video:
            return HandleExceptionCustomVideoDataset(image_paths,
                                      transform_img=self.transform_image(train=False, video=self.video),
                                      transform_vid=self.transformer_video(),
                                      clips_dictionary=dictionary_clips)
        else:
            return CustomImageDataset(image_paths,
                                      transform=transform)
        
    def get_true_labels(self, split, mode, dict_clips=None,video=False):
        if mode == 'prod':
            return None

        log.info(f"Path for feather {self.all_train_labels_path}")
        if split == 'train':
            df = pd.read_feather(self.all_train_labels_path)
        else:
            df = pd.read_feather(self.test_labels_path)
        
        # convert string labels to int labels
        mapped_label_col = df['class'].map(self.label_map)
        df['class'] = mapped_label_col

        # turn Dataframe into a dict
        df = df.set_index('id')
        labels_dict = df.to_dict()['class']

        # get a list of corresponding labels
        if split == 'train':
            image_names = self.get_unlabeled_image_names(dictionary_clips=dict_clips, video=video)
        else:
            image_names = self.get_evaluation_image_names(video=video)

        labels = [labels_dict[image_name] for image_name in image_names]
        return labels


class JPLRunner:
    def __init__(self, dataset_type, problem_type, dataset_dir, api_url, problem_task, team_secret, gov_team_secret,
                 data_paths, mode, simple_run, batch_size, testing=False):

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

        if self.video:
            if accelerator.is_local_main_process:
                # This line is a patch for a bug in the pytorch core code base: https://github.com/pytorch/pytorch/issues/61755
                torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
                initial_model = torch.hub.load("facebookresearch/pytorchvideo", 
                                            model='slowfast_r50', 
                                            pretrained=True,
                                            force_reload=True)
            accelerator.wait_for_everyone()
            self.initial_model = torch.hub.load("facebookresearch/pytorchvideo", 
                                            model='slowfast_r50', 
                                            pretrained=True)
            
        else:
            self.initial_model = models.resnet50(pretrained=True)
            self.initial_model.fc = torch.nn.Identity()

        self.testing = testing
        self.mode = mode
        self.simple_run = simple_run
        self.batch_size = batch_size
        
        if not os.path.exists('saved_vote_matrices') and accelerator.is_local_main_process:
            os.makedirs('saved_vote_matrices')
        accelerator.wait_for_everyone()
        self.vote_matrix_dict = {}
        self.vote_matrix_save_path = os.path.join('saved_vote_matrices',
                                                  datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

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
        self.jpl_storage.set_image_path(self.dataset_dir, current_dataset['name'], self.api.data_type, self.video)

    def run_checkpoints(self):
        try:
            self.run_checkpoints_base()
            self.run_checkpoints_adapt()
        except Exception as ex:
            self.api.deactivate_session(self.api.session_token)
            logging.exception('EXCEPTION has occured during joint training:')

    def run_checkpoints_base(self):
        log.info("Enter checkpoint")
        self.update_jpl_information()
        for i in range(self.num_base_checkpoints):
            self.run_one_checkpoint("Base", i)
            
    def run_checkpoints_adapt(self):
        self.update_jpl_information()
        for i in range(self.num_base_checkpoints):
            self.run_one_checkpoint("Adapt", i)

    def _get_weighted_dist(self, vote_matrix, weights, classes):
        weak_labels = []
        for row in vote_matrix:
            weak_label = np.zeros((len(classes),))
            for i in range(len(row)):
                weak_label[int(row[i])] += weights[i]
            weak_labels.append(weak_label / weak_label.sum())
        return weak_labels

    def run_one_checkpoint(self, phase, checkpoint_num):
        log.info('------------------------------------------------------------')
        log.info('--------------------{} Checkpoint: {}'.format(phase, checkpoint_num)+'---------------------')
        log.info('------------------------------------------------------------')

        start_time = time.time()

        #log.info(f"session STATUS {self.api.get_session_status()}")
        # Skip checkpoint before getting available budget
        

        available_budget = self.get_available_budget()
        if checkpoint_num == 0:
            self.jpl_storage.labeled_images, self.jpl_storage.dictionary_clips = \
                self.api.request_label(video=self.video)
            log.info(f'Get initial seeds at {checkpoint_num} checkpoint')
        elif 1 <= checkpoint_num <= 3:
            new_labeled_images, new_dictionary_clips = self.api.request_label(video=self.video)
            self.jpl_storage.add_labeled_images(new_labeled_images)
            if self.jpl_storage.dictionary_clips != None:
                self.jpl_storage.dictionary_clips.update(new_dictionary_clips)
            log.info(f'Get seeds at {checkpoint_num} checkpoints')

            # if checkpoint_num == 1:
            #     log.info('{} Skip Checkpoint: {} Elapsed Time =  {}'.format(phase,
            #                                                                 checkpoint_num,
            #                                                                 time.strftime("%H:%M:%S",
            #                                                                               time.gmtime(
            #                                                                                   time.time() - start_time))))
            #     return self.api.skip_checkpoint()
        
        # Get sets of unlabeled samples
        unlabeled_image_names = self.jpl_storage.get_unlabeled_image_names(self.jpl_storage.dictionary_clips,
                                                                           self.video)
        log.info('Number of unlabeled data: {}'.format(len(unlabeled_image_names)))

        
        
        if checkpoint_num >= 4:
            """ For the last evaluation we used to start asking for custom labels after the first 2 checkpoints.
            Moreover we adopted randomActive learning for the first query. Do we want it?
            """

            # Add all labeled data
            candidates = self.random_active_learning.find_candidates(available_budget, unlabeled_image_names)
            self.request_labels(candidates, self.video)

            # if checkpoint_num == 6:                
            #     log.info('{} Skip Checkpoint: {} Elapsed Time =  {}'.format(phase,
            #                                                     checkpoint_num,
            #                                                     time.strftime("%H:%M:%S",
            #                                                                     time.gmtime(time.time()-start_time))))
            #     return self.api.skip_checkpoint()

        labeled_dataset, val_dataset = self.jpl_storage.get_labeled_dataset(checkpoint_num, self.jpl_storage.dictionary_clips, self.video)
        #log.info(f"number training : {len(labeled_dataset.filepaths)}")
        unlabeled_train_dataset = self.jpl_storage.get_unlabeled_dataset(True, self.video, dict_clips=self.jpl_storage.dictionary_clips)
        #log.info(f" number unlabeled_train_dataset: {len(unlabeled_train_dataset.filepaths)}")
        unlabeled_test_dataset = self.jpl_storage.get_unlabeled_dataset(False, self.video, dict_clips=self.jpl_storage.dictionary_clips)
        #log.info(f"unlabeled_test_dataset: {len(unlabeled_test_dataset.filepaths)}")
        unlabeled_train_labels = self.jpl_storage.get_true_labels('train', self.mode, dict_clips=self.jpl_storage.dictionary_clips, video=self.video)
        #log.info(f"unlabeled_train_labels: {len(unlabeled_train_labels)}")
        

        if self.video:
            processed_scads_path = self.data_paths[2]
        else:
            processed_scads_path = self.data_paths[3]

        task = Task(self.jpl_storage.name,
                    labels_to_concept_ids(self.jpl_storage.classes),
                    (224, 224), 
                    labeled_dataset,
                    unlabeled_train_dataset,
                    val_dataset,
                    self.batch_size,
                    self.jpl_storage.whitelist,
                    self.data_paths[0],
                    self.data_paths[1],
                    processed_scads_path,
                    unlabeled_test_data=unlabeled_test_dataset,
                    unlabeled_train_labels=unlabeled_train_labels,
                    video_classification=self.video)
        task.set_initial_model(self.initial_model)
        controller = Controller(task, self.simple_run)
        
        if self.video:
            taglet_executor, taglets = controller.train_end_model()
            labeled = task.get_labeled_train_data()
            unlabeled_train = task.get_unlabeled_data(True) # augmentation is applied
            unlabeled_test = task.get_unlabeled_data(False)
            val = task.get_validation_data()

            if unlabeled_test is not None:
                log.info("Executing taglets on unlabeled data")
                self.unlabeled_vote_matrix = taglet_executor.execute(unlabeled_test, video=self.video)
                log.info("Finished executing taglets on unlabeled data")

                if task.unlabeled_train_labels is not None:
                    log.info('Accuracies of each taglet on the unlabeled train data:')
                    for i in range(len(taglets)):
                        acc = np.sum(self.unlabeled_vote_matrix[:, i] == np.array(task.unlabeled_train_labels)) / len(task.unlabeled_train_labels)
                        log.info("Module {} - acc {:.4f}".format(taglets[i].name, acc))

                # Combines taglets' votes into soft labels
                if val is not None and len(val) >= len(task.classes) * 10:
                    # Weight votes using development set
                    weights = [taglet.evaluate(val) for taglet in taglets]
                    log.info("Validation accuracies of each taglet:")
                    for w, taglet in zip(weights, taglets):
                        log.info("Module {} - acc {:.4f}".format(taglet.name, w))
                else:
                    # Weight all votes equally
                    weights = [1.0] * len(taglets)
                
                #log.info(f"Unlabeled vote matric: {self.unlabeled_vote_matrix} and shape {self.unlabeled_vote_matrix.shape}, weights: {weights}, task.classes {task.classes}")
                weak_labels = self._get_weighted_dist(self.unlabeled_vote_matrix, weights, task.classes)


                if task.unlabeled_train_labels is not None:
                    log.info('Accuracy of the labelmodel on the unlabeled train data:')
                    predictions = np.asarray([np.argmax(label) for label in weak_labels])
                    acc = np.sum(predictions == task.unlabeled_train_labels) / len(task.unlabeled_train_labels)
                    log.info('Acc {:.4f}'.format(acc))
            else:
                if val is not None and len(val) >= len(task.classes) * 10:
                    # Weight votes using development set
                    weights = [taglet.evaluate(val) for taglet in taglets]
                    log.info("Validation accuracies of each taglet:")
                    for w, taglet in zip(weights, taglets):
                        log.info("Module {} - acc {:.4f}".format(taglet.name, w))
                else:
                    # Weight all votes equally
                    weights = [1.0] * len(taglets)

            # Evaluate on test set
            evaluation_dataset = self.jpl_storage.get_evaluation_dataset(self.video)    

            log.info("Executing taglets on eval data")
            self.unlabeled_vote_matrix = taglet_executor.execute(evaluation_dataset, video=self.video)
            log.info("Finished executing taglets on eval data")

            log.info(f"Use weights of validation to weight labelers vote {weights}")
            weak_labels = self._get_weighted_dist(self.unlabeled_vote_matrix, weights, task.classes)
            predictions = np.asarray([np.argmax(label) for label in weak_labels])

            test_labels = self.jpl_storage.get_true_labels('test', self.mode, 
                                                            dict_clips=self.jpl_storage.dictionary_clips, 
                                                            video=self.video)
            if test_labels is not None:        
                log.info('Accuracy of Taglets on the eval data:')
                acc = np.sum(predictions == test_labels) / len(test_labels)
                log.info('Acc {:.4f}'.format(acc))

            #log.info(f"Predictions with one taglet: {predictions}")
            log.info(f"Length eval data: {len(evaluation_dataset.filepaths)}")

            prediction_names = []
            for p in predictions:
                prediction_names.append([k for k, v in self.jpl_storage.label_map.items() if v == p][0])

            #log.info(f"Predictions with one taglet: {prediction_names}, length: {len(prediction_names)}")
            evaluate_on = self.jpl_storage.get_evaluation_image_names(self.video)
            #log.info(f"Predictions with one taglet: {evaluate_on}, length: {len(evaluate_on)}")

            predictions_dict = {'id': self.jpl_storage.get_evaluation_image_names(self.video), 'class': prediction_names}

            self.submit_predictions(predictions_dict)

        else:
            end_model = controller.train_end_model()
            if self.vote_matrix_save_path is not None:
                val_vote_matrix, unlabeled_vote_matrix = controller.get_vote_matrix()
                if val_dataset is not None:
                    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
                    val_image_names = [os.path.basename(image_path) for image_path in val_dataset.filepaths]
                    val_labels = [image_labels for _, image_labels in val_loader]
                else:
                    val_image_names = None
                    val_labels = None
                checkpoint_dict = {'val_images_names': val_image_names,
                                   'val_images_votes': val_vote_matrix,
                                   'val_images_labels': val_labels,
                                   'unlabeled_images_names': self.jpl_storage.get_unlabeled_image_names(),
                                   'unlabeled_images_votes': unlabeled_vote_matrix}
                self.vote_matrix_dict[f'{phase} {checkpoint_num}'] = checkpoint_dict
                with open(self.vote_matrix_save_path, 'wb') as f:
                    pickle.dump(self.vote_matrix_dict, f)

            evaluation_dataset = self.jpl_storage.get_evaluation_dataset(self.video)
            outputs = end_model.predict(evaluation_dataset)
            predictions = np.argmax(outputs, 1)

            test_labels = self.jpl_storage.get_true_labels('test', self.mode)
            if test_labels is not None:
                log.info('Accuracy of taglets on this checkpoint:')
                acc = np.sum(predictions == test_labels) / len(test_labels)
                log.info('Acc {:.4f}'.format(acc))

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
        log.info('Done requesting labels!')
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
             mode, simple_run, batch_size):
    if problem_task == 'all':
        log.info('Execute all tasks')
        print(log.info('Execute all tasks'))
        jpl = JPL(api_url, team_secret, gov_team_secret, dataset_type)
        problem_task_list = jpl.get_available_tasks(problem_type)
        for task in problem_task_list:
            runner = JPLRunner(dataset_type, problem_type, dataset_dir, api_url, task, team_secret, gov_team_secret,
                               data_paths, mode, simple_run, batch_size, testing=False)
            runner.run_checkpoints()
    else:
        log.info("Execute a single task")
        runner = JPLRunner(dataset_type, problem_type, dataset_dir, api_url, problem_task, team_secret, gov_team_secret,
                           data_paths, mode, simple_run, batch_size, testing=False)
        runner.run_checkpoints()


def setup_production(simple_run):
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
    data_paths = ('/tmp/predefined/scads.spring2021.sqlite3',
                  '/tmp/predefined/embeddings/numberbatch-en19.08.txt.gz',
                  '/tmp/predefined/embeddings/spring2021_video_processed_numberbatch.h5',
                  '/tmp/predefined/embeddings/spring2021_processed_numberbatch.h5')

    if simple_run:
        log.info(f"Running production in simple mode, not all GPUs required")
    else:   
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
                        default="true",
                        help="Option to choose whether to execute or not the entire trining pipeline")
    parser.add_argument("--folder",
                        type=str,
                        default="external",# external, evaluation
                        help="Option to choose the data folder")
    parser.add_argument("--batch_size",
                        type=int,
                        default="4",
                        help="Universal batch size")
    args = parser.parse_args()
    

    if args.simple_version == 'true':
        simple_run = True
    else: 
        simple_run = False
    batch_size = args.batch_size

    if args.mode == 'prod':
        variables = setup_production(simple_run)
    else:
        variables = setup_development()
    mode = args.mode

    # Set Scads root path
    Scads.set_root_path(os.path.join(variables[2], 'external'))
    
    saved_api_response_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_api_response')
    if not os.path.exists(saved_api_response_dir) and accelerator.is_local_main_process:
        os.makedirs(saved_api_response_dir, exist_ok=True)
    accelerator.wait_for_everyone()
    
    dataset_type = variables[0]
    problem_type = variables[1]
    log.info(f"Problem type: {problem_type}")
    dataset_dir = os.path.join(variables[2], 'datasets', args.folder)
    log.info(f"Dataset dir: {dataset_dir}")
    api_url = variables[3]
    problem_task = variables[4]
    team_secret = variables[5]
    gov_team_secret = variables[6]
    data_paths = variables[7]
    

    valid_dataset_types = ['sample', 'full', 'all']
    if dataset_type not in valid_dataset_types:
        raise Exception(f'Invalid `dataset_type`, expected one of {valid_dataset_types}')

    # Check problem type is valid
    valid_problem_types = ['image_classification', 'object_detection', 'machine_translation', 'video_classification',
                           'all']
    if problem_type not in valid_problem_types:
        raise Exception(f'Invalid `problem_type`, expected one of {valid_problem_types}')

    # Check dataset directory exists
    log.info(f"Dataset dir: {dataset_dir}")
    if not Path(dataset_dir).exists():
        raise Exception('`dataset_dir` does not exist..')

    #_logger = logging.getLogger(__name__)
    #_logger.level = logging.DEBUG
    #stream_handler = logging.StreamHandler(sys.stdout)
    #stream_handler = logging.StreamHandler(logger.log)
    #formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    #stream_handler.setFormatter(formatter)
    #_logger.addHandler(stream_handler)
    
    workflow(dataset_type, problem_type, dataset_dir, api_url, problem_task, team_secret, gov_team_secret, data_paths,
             mode, simple_run, batch_size)
    

if __name__ == "__main__":
    main()
