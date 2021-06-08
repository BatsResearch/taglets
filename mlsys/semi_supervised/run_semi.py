import os
import time
import json
import logging
import argparse
#import requests
from pathlib import Path

from accelerate import Accelerator
accelerator = Accelerator()
import torch
import numpy as np
import pandas as pd
import torchvision.models as models
import torchvision.transforms as transforms

from taglets.task import Task
from taglets.data import CustomImageDataset, CustomVideoDataset
from taglets.controller import Controller
from taglets.task.utils import labels_to_concept_ids
from taglets.active import RandomActiveLearning, LeastConfidenceActiveLearning
from taglets.scads import Scads


gpu_list = os.getenv("LWLL_TA1_GPUS")
if gpu_list is not None and gpu_list != "all":
    gpu_list = [x for x in gpu_list.split(" ")]
    print(gpu_list)
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

        #self.team_secret = team_secret
        #self.gov_team_secret = gov_team_secret
        self.url = ROOT # root that contains the folder with info 
        self.session_token = ''
        self.data_type = dataset_type
        self.labels_path = ''
        self.saved_api_response_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_api_response')

    def get_available_tasks(self, problem_type):
        """
        Get all available tasks.
        :return: A list of tasks (problems)
        """
        #headers = {'user_secret': self.team_secret,
        #           'govteam_secret': self.gov_team_secret
        #           }
        r = json.load(open(self.url + '/list_task.json', 'r'))
        task_list = r['tasks']

        subset_tasks = []
        for _task in task_list:
            r = json.load(open(self.url + '/task_metadata.json', 'r'))
            task_metadata = r[_task]
            if task_metadata['task_metadata']['problem_type'] == problem_type:
                subset_tasks.append(_task)
        return subset_tasks

    def get_task_metadata(self, task_name):
        """
        Get metadata about a task.
        :param task_name: The name of the task (problem)
        :return: The task metadata
        """
        #headers = {'user_secret': self.team_secret,
        #           'govteam_secret': self.gov_team_secret}
        
        r = json.load(open(self.url + '/task_metadata.json', 'r'))
        task_meta = r[task_name]
        return task_meta['task_metadata']

    def create_session(self, task_name):
        """
        Create a new session.
        :param task_name: The name of the task (problem
        :return: None
        """
        #headers = {'user_secret': self.team_secret,
        #           'govteam_secret': self.gov_team_secret}
        #session_json = {'session_name': 'testing', 'data_type': self.data_type, 'task_id': task_name}
        
        response = json.load(open(self.url + '/create_session.json', 'r'))

        session_token = response[task_name]
        self.session_token = session_token

    def get_session_status(self):
        """
        Get the session status.
        :return: The session status
        """
        # headers = {'user_secret': self.team_secret,
        #            'govteam_secret': self.gov_team_secret,
        #            'session_token': self.session_token}
        r = json.load(open(self.url + "/session_status.json", 'r'))[self.session_token]
        if 'Session_Status' in r:
            return r['Session_Status']
        else:
            return {}

    def get_initial_seed_labels(self, video=False):
        """
        Get seed labels.
        :return: A list of lists with name and label e.g., ['2', '1.png'], ['7', '2.png'], etc.
        """

        log.info('Request seed labels.')
        # headers = {'user_secret': self.team_secret,
        #            'govteam_secret': self.gov_team_secret,
        #            'session_token': self.session_token}
        #  log.debug(f"HEADERS: {headers}")
        response = self.get_only_once("seed_labels")
        labels = response['Labels']

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
        if accelerator.is_local_main_process:
            headers_active_session = {'user_secret': self.team_secret,
                                      'govteam_secret': self.gov_team_secret,
                                      'session_token': self.session_token}
    
            r = requests.post(self.url + "/deactivate_session",
                              json={'session_token': deactivate_session},
                              headers=headers_active_session)

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
        log.debug(f"Query for new labels: {type(query['example_ids'][0])}")
        # headers = {'user_secret': self.team_secret,
        #            'govteam_secret': self.gov_team_secret,
        #            'session_token': self.session_token}
        response = self.post_only_once("query_labels", query)
        labels = response['Labels']
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
        predictions_json={'predictions': predictions}
        return self.post_only_once('submit_predictions', headers, predictions_json)
        
    def deactivate_all_sessions(self):
        headers_session = {'user_secret': self.team_secret,
                           'govteam_secret': self.gov_team_secret}
        r = requests.get(self.url + "/list_active_sessions", headers=headers_session)
        active_sessions = r.json()['active_sessions']
        for session_token in active_sessions:
            self.deactivate_session(session_token)

    def post_only_once(self, command, posting_json):

        if accelerator.is_local_main_process:
            # Get labels for query
            dataset = json.load(open(self.url + "/session_status.json", 'r'))[self.session_token]['name']
            df = pd.read_feather(f"{self.labels_path}/{dataset}/labels_{self.data_type}/labels_test.feather").set_index('id').loc[query['example_ids']]
            list_labels = []
            for row in df.iterrows():
                list_labels += [{'id':row[0],
                                'class':row[1]['class']}]  
            query_labels = {'Labels': list_labels}

            with open(os.path.join(self.saved_api_response_dir, command.replace("/", "_") + "_response.json"), "w") as f:
                json.dump(query_labels, f)
        accelerator.wait_for_everyone()
        with open(os.path.join(self.saved_api_response_dir, command.replace("/", "_") + "_response.json"), "r") as f:
            response = json.load(f)
        return response
    
    def get_only_once(self, command):
        if accelerator.is_local_main_process:
            r = json.load(open(self.url + '/' + command + '.json', 'r'))
            with open(os.path.join(self.saved_api_response_dir, command.replace("/", "_") + "_response.json"), "w") as f:
                json.dump(r[self.session_token], f)
        accelerator.wait_for_everyone()
        with open(os.path.join(self.saved_api_response_dir, command.replace("/", "_") + "_response.json"), "r") as f:
            response = json.load(f)
        return response
    



def main():
    parser = argparse.ArgumentParser('argument for training')
    args = parser.parse_args()

    log.info("Setting things up")

    # Generate checkpoint labels