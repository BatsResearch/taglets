import os
import sys
import yaml
import json
import random
import logging
import argparse
import requests
import numpy as np
import pandas as pd
from logging import StreamHandler
from pathlib import Path
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

from collections import defaultdict
import torch
from torch import nn
from accelerate import Accelerator
accelerator = Accelerator()

from .utils import Config
from .data import CustomDataset
from .modules import ClipBaseline, CoopBaseline, TptBaseline, VPTBaseline, \
                     AdjustAndAdapt, VPTPseudoBaseline, CoopPseudoBaseline, \
                     TeacherStudent, DisambiguateTeacherStudent, \
                     VPTPseudoDisambiguate, TwoStageClassifier

gpu_list = os.getenv("LWLL_TA1_GPUS")
if gpu_list is not None and gpu_list != "all":
    gpu_list = [x for x in gpu_list.split(" ")]
    print(gpu_list)
    os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(gpu_list)

logger_ = logging.getLogger()
logger_.level = logging.INFO
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')

class AccelerateHandler(StreamHandler):
    def __init__(self, stream):
        super().__init__(stream)
    
    def emit(self, record):
        if accelerator.is_local_main_process:
            super().emit(record)


stream_handler = AccelerateHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)
logger_.addHandler(stream_handler)

DEFAULT_TIMEOUT = 10 # seconds

log = logging.getLogger(__name__)

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


class ZSL_JPL:
    """
    A class to interact with ZSL_JPL-like APIs.
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
        #self.saved_api_response_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_api_response')
        
        retry_strategy = Retry(
            total=10,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = TimeoutHTTPAdapter(max_retries=retry_strategy)
        self.session = requests.Session()
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def create_session(self, task_name):
        """
        Create a new session.
        :param task_name: The name of the task (problem
        :return: None
        """
        headers = {'user_secret': self.team_secret,
                   'govteam_secret': self.gov_team_secret}
        session_json = {'session_name': 'testing', 'data_type': self.data_type, 
                        'task_id': task_name, "ZSL": True}
        
        response = self.post_only_once("auth/create_session", headers, session_json)
        log.debug(f"RESPONSE: {response}")
        session_token = response['session_token']
        self.session_token = session_token

    def get_session_status(self):
        """
        Get the session status.
        :return: The session status
        """
        headers = {'user_secret': self.team_secret,
                   'govteam_secret': self.gov_team_secret,
                   'session_token': self.session_token}
        r = self.get_only_once("session_status", headers)
        if 'Session_Status' in r:
            return r['Session_Status']
        else:
            return {}

    def get_seen_labeled_data(self):
        """
        Get seed labels.
        :return: A list of lists with name and label e.g., ['2', '1.png'], ['7', '2.png'], etc.
        """

        log.info('Request seen labeled data.')
        headers = {'user_secret': self.team_secret,
                   'govteam_secret': self.gov_team_secret,
                   'session_token': self.session_token}
        log.debug(f"HEADERS: {headers}")
        response = self.get_only_once("get_seen_labels", headers)
        labels = response['Labels']
        log.info(f"Number of labeled data: {len(labels)}")        
        seed_labels = [(image["id"], image["class"]) for image in labels]

        return seed_labels

    def get_unseen_unlabeled_data(self):
        """
        Get seed labels.
        :return: A list of lists with name and label e.g., ['2', '1.png'], ['7', '2.png'], etc.
        """

        log.info('Request unseen unlabeled data.')
        headers = {'user_secret': self.team_secret,
                   'govteam_secret': self.gov_team_secret,
                   'session_token': self.session_token}
        log.debug(f"HEADERS: {headers}")
        response = self.get_only_once("get_unseen_ids", headers)
        unlabeled_data = response['ids']
        log.info(f"Number of unlabeled examples: {len(unlabeled_data)}")        
        
        return unlabeled_data

    def submit_standard_zsl(self, predictions):
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
        predictions_json = {'predictions': predictions}
        return self.post_only_once('submit_standard_zsl_predictions', headers, predictions_json)

    def submit_generalized_zsl(self, predictions):
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
        predictions_json = {'predictions': predictions}
        log.info(f"Submit predictions...")
        return self.post_only_once('submit_predictions', headers, predictions_json)

    def post_only_once(self, command, headers, posting_json):
        if accelerator.is_local_main_process:
            r = self.session.post(self.url + "/" + command, json=posting_json, headers=headers)
            with open(os.path.join(command.replace("/", "_") + "_response.json"), "w") as f:
                json.dump(r.json(), f)
        accelerator.wait_for_everyone()
        with open(os.path.join(command.replace("/", "_") + "_response.json"), "r") as f:
            response = json.load(f)
        return response
    
    def get_only_once(self, command, headers):
        if accelerator.is_local_main_process:
            r = self.session.get(self.url + "/" + command, headers=headers)
            with open(os.path.join(command.replace("/", "_") + "_response.json"), "w") as f:
                json.dump(r.json(), f)
        accelerator.wait_for_everyone()
        with open(os.path.join(command.replace("/", "_") + "_response.json"), "r") as f:
            response = json.load(f)
        return response

    def deactivate_session(self, deactivate_session):
        if accelerator.is_local_main_process:
            headers_active_session = {'user_secret': self.team_secret,
                                      'govteam_secret': self.gov_team_secret,
                                      'session_token': self.session_token}
    
            r = self.session.post(self.url + "/deactivate_session",
                              json={'session_token': deactivate_session},
                              headers=headers_active_session)


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

    if simple_run:
        log.info(f"Running production in simple mode, not all GPUs required")
    else:   
        # check gpus are all
        if gpu_list != 'all':
            raise Exception(f'all gpus are required')
        
    return dataset_type, problem_type, dataset_dir, api_url, problem_task, team_secret, gov_team_secret


def setup_development():
    """
    This function returns the variables needed to launch the system in development.
    """

    # not sure this is very elegant. Let me know :)
    import dev_config

    return (dev_config.dataset_type, dev_config.problem_type, dev_config.dataset_dir, dev_config.api_url,
            dev_config.problem_task, dev_config.team_secret, dev_config.gov_team_secret)

def workflow(dataset_type, dataset_dir, api_url, 
             problem_task, team_secret, gov_team_secret, 
             obj_conf):
    
    api = ZSL_JPL(api_url, team_secret, gov_team_secret, dataset_type)
    
    # Create session
    api.create_session(problem_task)
    # Get session status
    session_status = api.get_session_status()
    if session_status == {}:
        raise Exception(f'The session status is empty: {session_status}')

    # Get dataset name
    dataset = session_status["current_dataset"]["name"]
    # Get class names of target task
    classes = session_status["current_dataset"]["classes"]
    seen_classes = session_status["current_dataset"]["seen_classes"]
    unseen_classes = session_status["current_dataset"]["unseen_classes"]
    # Create dict classes to pass as variable
    dict_classes = {'classes': classes,
                    'seen_classes': seen_classes,
                    'unseen_classes': unseen_classes}
    # Log number of classes
    log.info(f"Number of classes: {len(classes)}")
    log.info(f"Number of seen classes: {len(seen_classes)}")
    log.info(f"Number of unseen classes: {len(unseen_classes)}")
    # Get class descriptions
    zsl_descriptions = session_status["current_dataset"]["zsl_description"]
    unseen_descriptions = {}
    seen_descriptions = {}
    for k, desc in zsl_descriptions.items():
        if k in unseen_classes:
            unseen_descriptions[k] = desc
        elif k in seen_classes:
            seen_descriptions[k] = desc 

    # Path for images
    # dataset_dir = /users/cmenghin/data/bats/datasets/lwll/development/
    # dataset = UCMerced_LandUse
    # dataset_type = sample
    data_folder = f"{dataset_dir}/{dataset}/{dataset}_{dataset_type}"
    print(f"Data folder: {data_folder}")
    # Get labeled examples (seen classes)
    labeled_data = api.get_seen_labeled_data()
    # Get unlabeled examples (unseen classes)
    unlabeled_data = api.get_unseen_unlabeled_data()
    # Get test data (both seen and unseen classes)
    test_data = os.listdir(f"{data_folder}/test")


    # Create datasets
    labeled_files, labeles = zip(*labeled_data)
    label_to_idx = {c:idx for idx, c in enumerate(classes)}
    # Separate train and validation
    np.random.seed(obj_conf.validation_seed)
    train_indices = np.random.choice(range(len(labeled_files)),
                                    size=int(len(labeled_files)*obj_conf.ratio_train_val),
                                    replace=False)
    val_indices = list(set(range(len(labeled_files))).difference(set(train_indices)))
    
    train_labeled_files = np.array(labeled_files)[train_indices]
    train_labeles = np.array(labeles)[train_indices]

    val_labeled_files = np.array(labeled_files)[val_indices]
    val_labeles = np.array(labeles)[val_indices]

    # Training set (labeled seen): note that here tranform and augmentations are None.
    # These are attributes that everyone can set in the modules.
    train_seen_dataset = CustomDataset(train_labeled_files, data_folder, 
                                 transform=None, augmentations=None, 
                                 train=True, labels=train_labeles,
                                 label_map=label_to_idx)
    # Training set (unlabeled unseen)
    train_unseen_dataset = CustomDataset(unlabeled_data, data_folder, 
                                 transform=None, augmentations=None, 
                                 train=True, labels=None,
                                 label_map=label_to_idx)
    
    # Validation set (labeled seen)
    val_seen_dataset = CustomDataset(val_labeled_files, data_folder, 
                                 transform=None, augmentations=None, 
                                 train=True, labels=val_labeles,
                                 label_map=label_to_idx)
    # Test set (test seen and unseen)
    test_dataset = CustomDataset(test_data, data_folder, 
                                 transform=None, augmentations=None, 
                                 train=False, labels=None,
                                 label_map=label_to_idx)
    # Log info data
    log.info(f"Len training seen data: {len(train_seen_dataset.filepaths)}")
    log.info(f"Len training unseen data: {len(train_unseen_dataset.filepaths)}")
    log.info(f"Len validation seen data: {len(val_seen_dataset.filepaths)}")
    log.info(f"Len test data: {len(test_dataset.filepaths)}")
    # Define model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if obj_conf.MODEL == 'clip_baseline':
        log.info(f"The model in use is: {obj_conf.MODEL}")
        model = ClipBaseline(obj_conf, device=device, 
                             **dict_classes)
    elif obj_conf.MODEL == 'coop_baseline':
        log.info(f"The model in use is: {obj_conf.MODEL}")
        model = CoopBaseline(obj_conf, label_to_idx, 
                             device=device, 
                             **dict_classes) 

        val_accuracy, optimal_prompt = model.train(train_seen_dataset, val_seen_dataset, 
                                                   classes=seen_classes)
        log.info(f"Validation accuracy on seen classes: {val_accuracy}")
        log.info(f"The optimal prompt is {optimal_prompt}.")

        model.model.prefix = torch.nn.Parameter(torch.tensor(optimal_prompt[0]))
    elif obj_conf.MODEL == 'tpt_baseline':
        # TODO
        log.info(f"The model in use is: {obj_conf.MODEL}")
        model = TptBaseline(obj_conf, 
                             device=device, 
                             **dict_classes) 

    elif obj_conf.MODEL == 'vpt_baseline':
        log.info(f"The model in use is: {obj_conf.MODEL}")
        model = VPTBaseline(obj_conf, label_to_idx, 
                            device=device, 
                            **dict_classes)
        val_accuracy, optimal_prompt = model.train(train_seen_dataset, val_seen_dataset)
        log.info(f"Validation accuracy on seen classes: {val_accuracy}")
        log.info(f"The optimal prompt is {optimal_prompt}.")

        model.model.prefix = torch.nn.Parameter(torch.tensor(optimal_prompt[0]))
        model.model.image_pos_emb = torch.nn.Parameter(torch.tensor(optimal_prompt[1]))
    
    elif obj_conf.MODEL == 'vpt_pseudo_baseline':
        log.info(f"The model in use is: {obj_conf.MODEL}")
        model = VPTPseudoBaseline(obj_conf, label_to_idx, 
                            device=device, 
                            **dict_classes)
        val_accuracy, optimal_prompt = model.train(train_seen_dataset, val_seen_dataset,
                                                   train_unseen_dataset)
        log.info(f"Validation accuracy on seen classes: {val_accuracy}")
        log.info(f"The optimal prompt is {optimal_prompt}.")

    elif obj_conf.MODEL == 'coop_pseudo_baseline':
        log.info(f"The model in use is: {obj_conf.MODEL}")
        model = CoopPseudoBaseline(obj_conf, label_to_idx, 
                            device=device, 
                            **dict_classes)
        val_accuracy, optimal_prompt = model.train(train_seen_dataset, 
                                                   val_seen_dataset, classes=classes, 
                                                   unlabeled_data=train_unseen_dataset)
        log.info(f"Validation accuracy on seen classes: {val_accuracy}")
        log.info(f"The optimal prompt is {optimal_prompt}.")
    
    elif obj_conf.MODEL == 'vpt_pseudo_disambiguate':
        model = VPTPseudoDisambiguate(obj_conf, label_to_idx, 
                                      device=device, 
                                      **dict_classes)
        val_accuracy, optimal_prompt = model.train(train_seen_dataset, val_seen_dataset,
                                                   train_unseen_dataset)

    elif obj_conf.MODEL == 'teacher_student':
        log.info(f"The model in use is: {obj_conf.MODEL}")
        model = TeacherStudent(obj_conf, label_to_idx, 
                               data_folder, 
                               device=device, 
                               **dict_classes)
        val_accuracy, optimal_prompt = model.train(train_seen_dataset,
                                                   val_seen_dataset,
                                                   train_unseen_dataset)
        log.info(f"Validation accuracy on seen classes: {val_accuracy}")
        log.info(f"The optimal prompt is {optimal_prompt}.")

    elif obj_conf.MODEL == 'disambiguate_teacher_student':
        log.info(f"The model in use is: {obj_conf.MODEL}")
        model = DisambiguateTeacherStudent(obj_conf, label_to_idx, 
                               data_folder, 
                               device=device, 
                               **dict_classes)
        val_accuracy, optimal_prompt = model.train(train_seen_dataset,
                                                   val_seen_dataset,
                                                   train_unseen_dataset)
        log.info(f"Validation accuracy on seen classes: {val_accuracy}")
        log.info(f"The optimal prompt is {optimal_prompt}.")

    elif obj_conf.MODEL == 'adjust_and_adapt':
        log.info(f"The model in use is: {obj_conf.MODEL}")
        model = AdjustAndAdapt(obj_conf, label_to_idx, 
                               device=device, 
                               **dict_classes)
        vpt_prompts = model.train(train_labeled_files, 
                                  unlabeled_data, val_labeled_files,
                                  data_folder)
    elif obj_conf.MODEL == 'two_stage_classifier':
        log.info(f"The model in use is: {obj_conf.MODEL}")
        model = TwoStageClassifier(obj_conf, label_to_idx, 
                                device=device, 
                                **dict_classes)

        model.train(train_seen_dataset, train_unseen_dataset, val_seen_dataset)

    # Validate on test set (standard)
    std_predictions = model.test_predictions(test_dataset, 
                                             standard_zsl=True)
    # Submit predictions (standard)
    std_response = api.submit_standard_zsl(std_predictions.to_dict())
    log.info(f'Standard ZSL results: {std_response}')
    log.info(f"[STD] Unseen accuracy: {std_response['Session_Status']['standard_zsl_scores']['accuracy_unseen_std']}")
    log.info(f"[STD] Unseen average recall per class: {std_response['Session_Status']['standard_zsl_scores']['average_per_class_recall_unseen_std']}")
    
    # Validate on test set (general)
    gen_predictions = model.test_predictions(test_dataset,
                                             standard_zsl=False)
    # Submit predictions (general)
    gen_response = api.submit_generalized_zsl(gen_predictions.to_dict())
    log.info(f'Generalized ZSL results')
    log.info(f"[GEN] Accuracy all classes: {gen_response['Session_Status']['checkpoint_scores'][0]['accuracy_all_classes']}")
    log.info(f"[GEN] Accuracy seen classes: {gen_response['Session_Status']['checkpoint_scores'][0]['accuracy_seen']}")
    log.info(f"[GEN] Accuracy unseen classes: {gen_response['Session_Status']['checkpoint_scores'][0]['accuracy_unseen']}")
    


def main():
    parser = argparse.ArgumentParser(description="Run JPL task")
    parser.add_argument("--mode",
                        type=str,
                        default="prod",
                        help="The mode to execute the system. prod: system eval, dev: system development")
    parser.add_argument("--folder",
                        type=str,
                        default="evaluation",# development
                        help="Option to choose the data folder")
    parser.add_argument("--model_config",
                        type=str,
                        default="model_config.yml",
                        help="Name of model config file")
 
    args = parser.parse_args()

    if args.mode == 'prod':
        variables = setup_production(simple_run)
    else:
        variables = setup_development()
    mode = args.mode

    dataset_type = variables[0]
    problem_type = variables[1]
    log.info(f"Problem type: {problem_type}")
    dataset_dir = os.path.join(variables[2], args.folder)
    log.info(f"Dataset dir: {dataset_dir}")
    api_url = variables[3]
    problem_task = variables[4]
    team_secret = variables[5]
    gov_team_secret = variables[6]

    valid_dataset_types = ['sample', 'full']
    if dataset_type not in valid_dataset_types:
        raise Exception(f'Invalid `dataset_type`, expected one of {valid_dataset_types}')

    # Check problem type is valid
    valid_problem_types = ['image_classification']
    if problem_type not in valid_problem_types:
        raise Exception(f'Invalid `problem_type`, expected one of {valid_problem_types}')

    # Check dataset directory exists
    if not Path(dataset_dir).exists():
        print(dataset_dir)
        raise Exception('`dataset_dir` does not exist..')

    log.info(f"Current working directory: {os.getcwd()}")
    log.info(f"Input daata folder: {variables[2]}")
    with open(f'system/models_config/{args.model_config}', 'r') as file:
        config = yaml.safe_load(file)
    obj_conf = Config(config)


    # Set random seeds
    device = "cuda" if torch.cuda.is_available() else "cpu"
    np.random.seed(obj_conf.OPTIM_SEED)
    random.seed(obj_conf.OPTIM_SEED)
    torch.manual_seed(obj_conf.OPTIM_SEED)
    accelerator.wait_for_everyone()
    # Seed for cuda
    if torch.cuda.is_available():
        torch.cuda.manual_seed(obj_conf.OPTIM_SEED)
        torch.cuda.manual_seed_all(obj_conf.OPTIM_SEED)
        accelerator.wait_for_everyone()
    
    torch.backends.cudnn.benchmark = True


    workflow(dataset_type, dataset_dir, api_url, 
             problem_task, team_secret, gov_team_secret, obj_conf)

if __name__ == "__main__":
    main()