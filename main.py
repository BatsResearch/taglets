import argparse
import json
import logging
import os
import random
import sys
from collections import defaultdict
from logging import StreamHandler
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import scipy.stats as st
import torch
import yaml
from accelerate import Accelerator
from requests.adapters import HTTPAdapter
from torch import nn
from urllib3.util import Retry

accelerator = Accelerator()

from data import CustomDataset, dataset_custom_prompts
from methods import (
    ClipBaseline,
    CoopBaseline,
    CoopPseudoBaseline,
    TeacherStudent,
    VPTBaseline,
    VPTPseudoBaseline,
)
from utils import (
    Config,
    dataset_object,
    evaluate_predictions,
    get_class_names,
    get_labeled_and_unlabeled_data,
)



logger_ = logging.getLogger()
logger_.level = logging.INFO
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")


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

log = logging.getLogger(__name__)


def workflow(dataset_dir, obj_conf):
    # Get dataset name
    # We get the dataset name from the dev_config.py
    dataset = obj_conf.DATASET_NAME
    # Get class names of target task
    # define function for each dataset
    classes, seen_classes, unseen_classes = get_class_names(dataset, dataset_dir, obj_conf.SPLIT_SEED)
    # Create dict classes to pass as variable
    dict_classes = {
        "classes": classes,
        "seen_classes": seen_classes,
        "unseen_classes": unseen_classes,
    }
    # Log number of classes
    log.info(f"Number of classes split {obj_conf.SPLIT_SEED}: {len(classes)}")
    log.info(f"Number of seen classes split {obj_conf.SPLIT_SEED}: {len(seen_classes)}")
    log.info(f"List of seen classes split {obj_conf.SPLIT_SEED}: {seen_classes}\n")
    log.info(f"Number of unseen classes split {obj_conf.SPLIT_SEED}: {len(unseen_classes)}")
    log.info(f"List of unseen classes split {obj_conf.SPLIT_SEED}: {unseen_classes}\n")

    # Path for images
    # dataset_dir = /users/cmenghin/data/bats/datasets/
    # dataset = aPY
    data_folder = f"{dataset_dir}/{dataset}"
    print(f"Data folder: {data_folder}")
    # Get labeled examples (seen classes)
    # Get unlabeled examples (unseen classes)
    # Get test data (both seen and unseen classes)
    labeled_data, unlabeled_data, test_data = get_labeled_and_unlabeled_data(
        dataset, data_folder, seen_classes, unseen_classes, classes
    )

    # Create datasets
    labeled_files, labeles = zip(*labeled_data)
    unseen_labeled_files, unseen_labeles = zip(*unlabeled_data)
    test_labeled_files, test_labeles = zip(*test_data)
    label_to_idx = {c: idx for idx, c in enumerate(classes)}

    # Separate train and validation
    np.random.seed(obj_conf.validation_seed)
    train_indices = np.random.choice(
        range(len(labeled_files)),
        size=int(len(labeled_files) * obj_conf.ratio_train_val),
        replace=False,
    )
    val_indices = list(set(range(len(labeled_files))).difference(set(train_indices)))

    train_labeled_files = np.array(labeled_files)[train_indices]
    train_labeles = np.array(labeles)[train_indices]

    val_labeled_files = np.array(labeled_files)[val_indices]
    val_labeles = np.array(labeles)[val_indices]

    DatasetObject = dataset_object(obj_conf.DATASET_NAME)
    # Training set (labeled seen): note that here tranform and augmentations are None.
    # These are attributes that everyone can set in the modules.
    train_seen_dataset = DatasetObject(
        train_labeled_files,
        data_folder,
        transform=None,
        augmentations=None,
        train=True,
        labels=train_labeles,
        label_map=label_to_idx,
    )
    # Training set (unlabeled unseen)
    train_unseen_dataset = DatasetObject(
        unseen_labeled_files,
        data_folder,
        transform=None,
        augmentations=None,
        train=True,
        labels=None,
        label_map=label_to_idx,
    )

    # Validation set (labeled seen)
    val_seen_dataset = DatasetObject(
        val_labeled_files,
        data_folder,
        transform=None,
        augmentations=None,
        train=True,
        labels=val_labeles,
        label_map=label_to_idx,
    )
    # Test set (test seen and unseen)
    test_dataset = DatasetObject(
        test_labeled_files,
        data_folder,
        transform=None,
        augmentations=None,
        train=False,
        labels=None,
        label_map=label_to_idx,
    )
    # Log info data
    log.info(f"Len training seen data: {len(train_seen_dataset.filepaths)}")
    log.info(f"Len training unseen data: {len(train_unseen_dataset.filepaths)}")
    log.info(f"Len validation seen data: {len(val_seen_dataset.filepaths)}")
    log.info(f"Len test data: {len(test_dataset.filepaths)}")
    # Define model
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if obj_conf.MODEL == "clip_baseline":
        log.info(f"The model in use is: {obj_conf.MODEL}")
        model = ClipBaseline(obj_conf, label_to_idx, device=device, **dict_classes)
    elif obj_conf.MODEL == "coop_baseline":
        log.info(f"The model in use is: {obj_conf.MODEL}")
        model = CoopBaseline(obj_conf, label_to_idx, device=device, **dict_classes)

        val_accuracy, optimal_prompt = model.train(
            train_seen_dataset, val_seen_dataset, classes=seen_classes
        )
        log.info(f"Validation accuracy on seen classes: {val_accuracy}")
        log.info(f"The optimal prompt is {optimal_prompt}.")

        model.model.prefix = torch.nn.Parameter(torch.tensor(optimal_prompt[0]))

    elif obj_conf.MODEL == "tpt_baseline":
        # TODO
        log.info(f"The model in use is: {obj_conf.MODEL}")
        model = TptBaseline(obj_conf, device=device, **dict_classes)

    elif obj_conf.MODEL == "vpt_baseline":
        log.info(f"The model in use is: {obj_conf.MODEL}")
        model = VPTBaseline(obj_conf, label_to_idx, device=device, **dict_classes)
        val_accuracy, optimal_prompt = model.train(
            train_seen_dataset, val_seen_dataset, only_seen=True
        )
        log.info(f"Validation accuracy on seen classes: {val_accuracy}")
        log.info(f"The optimal prompt is {optimal_prompt}.")

        model.model.prefix = torch.nn.Parameter(torch.tensor(optimal_prompt[0]))
        model.model.image_pos_emb = torch.nn.Parameter(torch.tensor(optimal_prompt[1]))

    elif obj_conf.MODEL == "vpt_pseudo_baseline":
        log.info(f"The model in use is: {obj_conf.MODEL}")
        model = VPTPseudoBaseline(obj_conf, label_to_idx, device=device, **dict_classes)
        val_accuracy, optimal_prompt = model.train(
            train_seen_dataset, val_seen_dataset, train_unseen_dataset
        )
        log.info(f"Validation accuracy on seen classes: {val_accuracy}")
        log.info(f"The optimal prompt is {optimal_prompt}.")

    elif obj_conf.MODEL == "coop_pseudo_baseline":
        log.info(f"The model in use is: {obj_conf.MODEL}")
        model = CoopPseudoBaseline(
            obj_conf, label_to_idx, device=device, **dict_classes
        )
        val_accuracy, optimal_prompt = model.train(
            train_seen_dataset,
            val_seen_dataset,
            classes=classes,
            unlabeled_data=train_unseen_dataset,
        )
        log.info(f"Validation accuracy on seen classes: {val_accuracy}")
        log.info(f"The optimal prompt is {optimal_prompt}.")

    elif obj_conf.MODEL == "vpt_pseudo_disambiguate":
        model = VPTPseudoDisambiguate(
            obj_conf, label_to_idx, device=device, **dict_classes
        )
        val_accuracy, optimal_prompt = model.train(
            train_seen_dataset, val_seen_dataset, train_unseen_dataset
        )

    elif obj_conf.MODEL == "teacher_student":
        log.info(f"The model in use is: {obj_conf.MODEL}")
        model = TeacherStudent(
            obj_conf, label_to_idx, data_folder, device=device, **dict_classes
        )
        val_accuracy, optimal_prompt = model.train(
            train_seen_dataset,
            val_seen_dataset,
            train_unseen_dataset,
            test_dataset,
            test_labeled_files,
            test_labeles,
        )
        log.info(f"Validation accuracy on seen classes: {val_accuracy}")
        log.info(f"The optimal prompt is {optimal_prompt}.")

    elif obj_conf.MODEL == "disambiguate_teacher_student":
        log.info(f"The model in use is: {obj_conf.MODEL}")
        model = DisambiguateTeacherStudent(
            obj_conf, label_to_idx, data_folder, device=device, **dict_classes
        )
        val_accuracy, optimal_prompt = model.train(
            train_seen_dataset, val_seen_dataset, train_unseen_dataset
        )
        log.info(f"Validation accuracy on seen classes: {val_accuracy}")
        log.info(f"The optimal prompt is {optimal_prompt}.")

    elif obj_conf.MODEL == "adjust_and_adapt":
        log.info(f"The model in use is: {obj_conf.MODEL}")
        model = AdjustAndAdapt(obj_conf, label_to_idx, device=device, **dict_classes)
        vpt_prompts = model.train(
            train_labeled_files, unlabeled_data, val_labeled_files, data_folder
        )
    elif obj_conf.MODEL == "two_stage_classifier":
        log.info(f"The model in use is: {obj_conf.MODEL}")
        model = TwoStageClassifier(
            obj_conf, label_to_idx, device=device, **dict_classes
        )

        model.train(train_seen_dataset, train_unseen_dataset, val_seen_dataset)

    # Validate on test set (standard)
    std_predictions = model.test_predictions(test_dataset, standard_zsl=True)
    # Submit predictions (standard)
    std_response = evaluate_predictions(
        std_predictions,
        test_labeled_files,
        test_labeles,
        unseen_classes,
        standard_zsl=True,
    )
    log.info(f"ZSL accuracy: {std_response}")

    # Validate on test set (general)
    gen_predictions = model.test_predictions(test_dataset, standard_zsl=False)
    # Submit predictions (general)
    unseen_accuracy, seen_accuracy, harmonic_mean = evaluate_predictions(
        gen_predictions,
        test_labeled_files,
        test_labeles,
        unseen_classes,
        seen_classes,
        standard_zsl=False,
    )
    log.info(f"Generalized ZSL results")
    log.info(f"Accuracy seen classes: {seen_accuracy}")
    log.info(f"Accuracy unseen classes: {unseen_accuracy}")
    log.info(f"Harmonic mean: {harmonic_mean}")

    # Store model results
    store_results(obj_conf, std_response, unseen_accuracy, seen_accuracy, harmonic_mean)

 
def main():
    parser = argparse.ArgumentParser(description="Run JPL task")
    parser.add_argument(
        "--model_config",
        type=str,
        default="model_config.yml",
        help="Name of model config file",
    )

    args = parser.parse_args()

    log.info(f"Current working directory: {os.getcwd()}")

    with open(f"methods_config/{args.model_config}", "r") as file:
        config = yaml.safe_load(file)

    # Transform configs to object
    obj_conf = Config(config)

    # Declare seed
    optim_seed = int(os.environ["OPTIM_SEED"])
    obj_conf.OPTIM_SEED = optim_seed
    # Define backbone
    obj_conf.VIS_ENCODER = os.environ["VIS_ENCODER"]
    # Define dataset name
    obj_conf.DATASET_NAME = os.environ["DATASET_NAME"]
    # Define split seed
    obj_conf.SPLIT_SEED = int(os.environ["SPLIT_SEED"])
    # Define dataset's template for textual prompts
    obj_conf.PROMPT_TEMPLATE = dataset_custom_prompts[obj_conf.DATASET_NAME]
    # Define data dir
    dataset_dir = obj_conf.DATASET_DIR
    log.info(f"Dataset dir: {dataset_dir}")

    # Check dataset directory exists
    if not Path(dataset_dir).exists():
        print(dataset_dir)
        raise Exception("`dataset_dir` does not exist..")

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

    workflow(dataset_dir, obj_conf)


if __name__ == "__main__":
    main()
