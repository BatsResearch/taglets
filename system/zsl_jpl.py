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
import scipy.stats as st
from logging import StreamHandler
from pathlib import Path
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

from collections import defaultdict
import torch
from torch import nn
from accelerate import Accelerator
accelerator = Accelerator()

from .utils import Config, get_class_names, get_labeled_and_unlabeled_data
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

def setup_development():
    """
    This function returns the variables needed to launch the system in development.
    """

    # not sure this is very elegant. Let me know :)
    import dev_config

    return (dev_config.dataset_dir, dev_config.dataset_name)

def workflow(dataset_dir, dataset_name, 
             obj_conf):
    
    # Get dataset name
    # We get the dataset name from the dev_config.py
    dataset = dataset_name
    # Get class names of target task
    # define function for each dataset
    classes, seen_classes, unseen_classes = get_class_names(dataset, 
                                                            dataset_dir)
    # Create dict classes to pass as variable
    dict_classes = {'classes': classes,
                    'seen_classes': seen_classes,
                    'unseen_classes': unseen_classes}
    # Log number of classes
    log.info(f"Number of classes: {len(classes)}")
    log.info(f"Number of seen classes: {len(seen_classes)}")
    log.info(f"Number of unseen classes: {len(unseen_classes)}")

    # Path for images
    # dataset_dir = /users/cmenghin/data/bats/datasets/
    # dataset = aPY
    data_folder = f"{dataset_dir}/{dataset}/gbu"
    print(f"Data folder: {data_folder}")
    # Get labeled examples (seen classes)
    # Get unlabeled examples (unseen classes)
    # Get test data (both seen and unseen classes)
    labeled_data, unlabeled_data, test_data = get_labeled_and_unlabeled_data(dataset, data_folder, 
                                                                             seen_classes, unseen_classes)


    # Create datasets
    labeled_files, labeles = zip(*labeled_data)
    unseen_labeled_files, unseen_labeles = zip(*unlabeled_data)
    test_labeled_files, test_labeles = zip(*test_data)
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
    train_unseen_dataset = CustomDataset(unseen_labeled_files, data_folder, 
                                 transform=None, augmentations=None, 
                                 train=True, labels=None,
                                 label_map=label_to_idx)
    
    # Validation set (labeled seen)
    val_seen_dataset = CustomDataset(val_labeled_files, data_folder, 
                                 transform=None, augmentations=None, 
                                 train=True, labels=val_labeles,
                                 label_map=label_to_idx)
    # Test set (test seen and unseen)
    test_dataset = CustomDataset(test_labeled_files, data_folder, 
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
        model = ClipBaseline(obj_conf, label_to_idx, device=device, 
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
    std_response = evaluate_predictions(std_predictions, test_labeles, 
                                        unseen_classes, standard_zsl=True)
    log.info(f"ZSL accuracy: {std_response}")
    
    # Validate on test set (general)
    gen_predictions = model.test_predictions(test_dataset, 
                                             standard_zsl=False)
    # Submit predictions (general)
    unseen_accuracy, seen_accuracy, harmonic_mean = evaluate_predictions(gen_predictions, test_labeles, 
                                                                         unseen_classes, seen_classes, 
                                                                         standard_zsl=False)
    log.info(f'Generalized ZSL results')
    log.info(f"Accuracy seen classes: {seen_accuracy}")
    log.info(f"Accuracy unseen classes: {unseen_accuracy}")
    log.info(f"Harmonic mean: {harmonic_mean}")
    
    # Store model results
    store_results(obj_conf, std_response, gen_response)

def evaluate_predictions(preds, labels, 
                         unseen_classes, seen_classes=None, standard_zsl=False):
    
    if standard_zsl:
        df_predictions['true'] = labels
        df_predictions = df_predictions[df_predictions['class'].isin(unseen_classes)]
        accuracy = np.sum(df_predictions['preds'] == df_predictions['true']) / df_predictions.shape[0]

        return accuracy
        
    else:
        df_predictions['true'] = labels
        unseen_predictions = df_predictions[df_predictions['class'].isin(unseen_classes)]
        unseen_accuracy = np.sum(unseen_predictions['preds'] == unseen_predictions['true']) / unseen_predictions.shape[0]

        seen_predictions = df_predictions[df_predictions['class'].isin(seen_classes)]
        seen_accuracy = np.sum(seen_predictions['preds'] == seen_predictions['true']) / seen_predictions.shape[0]

        harmonic_mean = st.hmean(seen_predictions, seen_accuracy)

        return unseen_accuracy, seen_accuracy, harmonic_mean

def main():
    parser = argparse.ArgumentParser(description="Run JPL task")
    parser.add_argument("--dataset",
                        type=str,
                        help="Name dataset")
    parser.add_argument("--model_config",
                        type=str,
                        default="model_config.yml",
                        help="Name of model config file")
 
    args = parser.parse_args()


    variables = setup_development()

    dataset_name = args.dataset
    dataset_dir = variables[2]
    log.info(f"Dataset dir: {dataset_dir}")
    
    # Check dataset directory exists
    if not Path(dataset_dir).exists():
        print(dataset_dir)
        raise Exception('`dataset_dir` does not exist..')

    log.info(f"Current working directory: {os.getcwd()}")
    log.info(f"Input data folder: {variables[2]}")
    with open(f'system/models_config/{args.model_config}', 'r') as file:
        config = yaml.safe_load(file)
    obj_conf = Config(config)

    optim_seed = int(os.environ["OPTIM_SEED"])
    obj_conf.OPTIM_SEED  = optim_seed

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


    workflow(dataset_dir, dataset_name, 
             obj_conf)

if __name__ == "__main__":
    main()