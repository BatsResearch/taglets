import argparse
import json
import logging
import os
import pickle
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
    AblationTeacherStudent,
    AllVPTPseudoBaseline,
    ClipBaseline,
    CoopBaseline,
    CoopPseudoBaseline,
    InitVPTBaseline,
    IterativeFixedPseudo,
    PostVPTBaseline,
    QuantileCoopPseudoBaseline,
    QuantileVPTPseudoBaseline,
    RankVPTBaseline,
    SeparateVPTBaseline,
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
    save_parameters,
    store_results,
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
    log.info(f"\n----------------------DATA INFO-----------------------\n")
    log.info(f"Number of classes split {obj_conf.SPLIT_SEED}: {len(classes)}")
    log.info(f"Number of seen classes split {obj_conf.SPLIT_SEED}: {len(seen_classes)}")
    log.info(f"Number of unseen classes split {obj_conf.SPLIT_SEED}: {len(unseen_classes)}")
    # Path for images
    data_folder = f"{dataset_dir}/{dataset}"
    log.info(f"Data folder: {data_folder}")
    log.info(f"\n-------------------------------------------------------------\n")
    
    # Get labeled data (seen classes)
    # Get unlabeled data (unseen classes)
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
    log.info(f"\n----------------------TRAINING DATA INFO-----------------------\n")
    log.info(f"Len training seen data: {len(train_seen_dataset.filepaths)}")
    log.info(f"Average number of labeled images per seen class:{len(train_seen_dataset.filepaths)/len(seen_classes)} ")
    log.info(f"Len training unseen data: {len(train_unseen_dataset.filepaths)}")
    log.info(f"Len validation seen data: {len(val_seen_dataset.filepaths)}")
    log.info(f"Len test data: {len(test_dataset.filepaths)}")
    log.info(f"\n-------------------------------------------------------------\n")
    # Define model
    device = "cuda" if torch.cuda.is_available() else "cpu"

    log.info(f"\n----------------------MODEL INFO-----------------------\n")
    if obj_conf.MODEL == "clip_baseline":
        log.info(f"The model in use is: {obj_conf.MODEL}")
        model = ClipBaseline(obj_conf, label_to_idx, device=device, **dict_classes)
    
    elif obj_conf.MODEL == "coop_baseline":
        log.info(f"The model in use is: {obj_conf.MODEL}")
        model = CoopBaseline(obj_conf, label_to_idx, device=device, **dict_classes)

        val_accuracy, optimal_prompt = model.train(
            train_seen_dataset, val_seen_dataset, classes=seen_classes
        )
        
        model.model.prefix = torch.nn.Parameter(torch.tensor(optimal_prompt[0]))

    elif obj_conf.MODEL == "vpt_baseline":
        log.info(f"The model in use is: {obj_conf.MODEL}")
        model = VPTBaseline(obj_conf, label_to_idx, device=device, **dict_classes)
        val_accuracy, optimal_prompt = model.train(
            train_seen_dataset, val_seen_dataset, only_seen=True
        )

        model.model.prefix = torch.nn.Parameter(torch.tensor(optimal_prompt[0]))

    elif obj_conf.MODEL == "no_label_vpt_baseline":
        log.info(f"The model in use is: {obj_conf.MODEL}")
        model = VPTBaseline(obj_conf, label_to_idx, device=device, **dict_classes)
        val_accuracy, optimal_prompt = model.train(
            train_seen_dataset, val_seen_dataset, only_seen=True
        )

        learned_prefix = torch.tensor(optimal_prompt[0]).to(device)
        log.info(f"Let's now train on the unseen classes exploiting learned prompts")
        model = InitVPTBaseline(
            obj_conf, 
            label_to_idx, 
            init_param=learned_prefix,
            device=device, 
            **dict_classes
        )
        val_accuracy, optimal_prompt = model.train(
            train_seen_dataset, val_seen_dataset, train_unseen_dataset
        )
        model.model.prefix = torch.nn.Parameter(torch.tensor(optimal_prompt[0]))

    elif obj_conf.MODEL == "mix_vpt_baseline":
        log.info(f"The model in use is: {obj_conf.MODEL}")

        filename = f"trained_prompts/{obj_conf.DATASET_NAME}_{obj_conf.MODEL}_{obj_conf.VIS_ENCODER.replace('/','')}_opt_{obj_conf.OPTIM_SEED}_spl_{obj_conf.SPLIT_SEED}.pickle"
        if os.path.exists(filename):
            with open(filename, "rb") as f:
                trained_prompts = pickle.load(f)
            learned_prefix = torch.tensor(trained_prompts[0]).to(device)
        else:
            model = VPTBaseline(obj_conf, label_to_idx, device=device, **dict_classes)
            val_accuracy, optimal_prompt = model.train(
                train_seen_dataset, val_seen_dataset, only_seen=True
            )
            save_parameters(optimal_prompt, obj_conf, init_seen=True)
            
            learned_prefix = torch.tensor(optimal_prompt[0]).to(device)
        
        log.info(f"Let's now train on the unseen classes exploiting learned prompts")
        model = InitVPTBaseline(
            obj_conf, 
            label_to_idx, 
            init_param=learned_prefix,
            kind='mix',
            device=device, 
            **dict_classes
        )
        val_accuracy, optimal_prompt = model.train(
            train_seen_dataset, val_seen_dataset, train_unseen_dataset
        )
        model.model.prefix = torch.nn.Parameter(torch.tensor(optimal_prompt[0]))

    elif obj_conf.MODEL == "cat_vpt_baseline":
        log.info(f"The model in use is: {obj_conf.MODEL}")

        filename = f"trained_prompts/{obj_conf.DATASET_NAME}_{obj_conf.MODEL}_{obj_conf.VIS_ENCODER.replace('/','')}_opt_{obj_conf.OPTIM_SEED}_spl_{obj_conf.SPLIT_SEED}.pickle"
        if os.path.exists(filename):
            with open(filename, "rb") as f:
                trained_prompts = pickle.load(f)
            learned_prefix = torch.tensor(trained_prompts[0]).to(device)
        else:
            model = VPTBaseline(obj_conf, label_to_idx, device=device, **dict_classes)
            val_accuracy, optimal_prompt = model.train(
                train_seen_dataset, val_seen_dataset, only_seen=True
            )
            save_parameters(optimal_prompt, obj_conf, init_seen=True)
            
            learned_prefix = torch.tensor(optimal_prompt[0]).to(device)
        
        log.info(f"Let's now train on the unseen classes exploiting learned prompts")
        model = InitVPTBaseline(
            obj_conf, 
            label_to_idx, 
            init_param=learned_prefix,
            kind='cat',
            device=device, 
            **dict_classes
        )
        val_accuracy, optimal_prompt = model.train(
            train_seen_dataset, val_seen_dataset, train_unseen_dataset
        )
        model.model.prefix = torch.nn.Parameter(torch.tensor(optimal_prompt[0]))

    elif obj_conf.MODEL == "combo_vpt_baseline":
        log.info(f"The model in use is: {obj_conf.MODEL}")

        filename = f"trained_prompts/{obj_conf.DATASET_NAME}_mix_vpt_baseline_{obj_conf.VIS_ENCODER.replace('/','')}_opt_{obj_conf.OPTIM_SEED}_spl_{obj_conf.SPLIT_SEED}.pickle"
        log.info(f"{filename}")
        if os.path.exists(filename):
            with open(filename, "rb") as f:
                trained_prompts = pickle.load(f)
            learned_prefix = torch.tensor(trained_prompts[0]).to(device)
        else:
            raise Exception(f"Sorry, no model found. Run mix_vpt with for the seen initialization.")

        log.info(f"Let's now train on the unseen classes exploiting learned prompts")

        filename = f"trained_prompts/{obj_conf.DATASET_NAME}_mix_vpt_baseline_{obj_conf.VIS_ENCODER.replace('/','')}_alpha_{1.0}_opt_{obj_conf.OPTIM_SEED}_spl_{obj_conf.SPLIT_SEED}.pickle"
        log.info(f"{filename}")
        if os.path.exists(filename):
            with open(filename, "rb") as f:
                unseen_trained_prompts = pickle.load(f)
            unseen_learned_prefix = torch.tensor(unseen_trained_prompts[0]).to(device)
        else:
            raise Exception(f"Sorry, no model found. Run mix_vpt with alpha=1")

        log.info(f"Shape seen prompt: {learned_prefix.shape} and alpha: {obj_conf.ALPHA}")
        log.info(f"Shape unseen prompt: {unseen_learned_prefix.shape} and alpha {obj_conf.ALPHA}")
        optimal_prompt = obj_conf.ALPHA*unseen_learned_prefix + (1 - obj_conf.ALPHA)*learned_prefix
        model = VPTBaseline(obj_conf, label_to_idx, device=device, **dict_classes)
        model.define_model()
        model.model.prefix = torch.nn.Parameter(optimal_prompt)

    elif obj_conf.MODEL == "after_combo_vpt_baseline":
        log.info(f"The model in use is: {obj_conf.MODEL}")

        filename = f"trained_prompts/{obj_conf.DATASET_NAME}_mix_vpt_baseline_{obj_conf.VIS_ENCODER.replace('/','')}_opt_{obj_conf.OPTIM_SEED}_spl_{obj_conf.SPLIT_SEED}.pickle"
        log.info(f"{filename}")
        if os.path.exists(filename):
            with open(filename, "rb") as f:
                trained_prompts = pickle.load(f)
            learned_prefix = torch.tensor(trained_prompts[0]).to(device)
        else:
            raise Exception(f"Sorry, no model found. Run mix_vpt with for the seen initialization.")

        log.info(f"Let's now train on the unseen classes exploiting learned prompts")

        filename = f"trained_prompts/{obj_conf.DATASET_NAME}_mix_vpt_baseline_{obj_conf.VIS_ENCODER.replace('/','')}_alpha_{1.0}_opt_{obj_conf.OPTIM_SEED}_spl_{obj_conf.SPLIT_SEED}.pickle"
        log.info(f"{filename}")
        if os.path.exists(filename):
            with open(filename, "rb") as f:
                unseen_trained_prompts = pickle.load(f)
            unseen_learned_prefix = torch.tensor(unseen_trained_prompts[0]).to(device)
        else:
            raise Exception(f"Sorry, no model found. Run mix_vpt with alpha=1")

        log.info(f"Shape seen prompt: {learned_prefix.shape} and alpha: {obj_conf.ALPHA}")
        log.info(f"Shape unseen prompt: {unseen_learned_prefix.shape} and alpha {obj_conf.ALPHA}")
        # When defining the model we give both seen and unseen prompts in input
        model = SeparateVPTBaseline(obj_conf,
            label_to_idx, 
            device=device, 
            seen_param=learned_prefix,
            unseen_param=unseen_learned_prefix,
            **dict_classes
        )

    elif obj_conf.MODEL == "post_vpt_baseline":
        log.info(f"The model in use is: {obj_conf.MODEL}")
        
        filename = f"trained_prompts/{obj_conf.DATASET_NAME}_{obj_conf.MODEL}_{obj_conf.VIS_ENCODER.replace('/','')}_opt_{obj_conf.OPTIM_SEED}_spl_{obj_conf.SPLIT_SEED}.pickle"
        if os.path.exists(filename):
            with open(filename, "rb") as f:
                trained_prompts = pickle.load(f)
            learned_prefix = torch.tensor(trained_prompts[0]).to(device)
        else:
            model = VPTBaseline(obj_conf, label_to_idx, device=device, **dict_classes)
            val_accuracy, optimal_prompt = model.train(
                train_seen_dataset, val_seen_dataset, only_seen=True
            )
            save_parameters(optimal_prompt, obj_conf, init_seen=True)
            
            learned_prefix = torch.tensor(optimal_prompt[0]).to(device)

        log.info(f"Let's now train on the unseen classes exploiting learned prompts")
        model = PostVPTBaseline(
            obj_conf, 
            label_to_idx, 
            init_param=learned_prefix,
            device=device, 
            **dict_classes
        )
        val_accuracy, optimal_prompt = model.train(
            train_seen_dataset, val_seen_dataset, train_unseen_dataset
        )
        
        model.model.prefix = torch.nn.Parameter(torch.tensor(optimal_prompt[0]))

    elif obj_conf.MODEL == 'rank_vpt_baseline':
        log.info(f"The model in use is: {obj_conf.MODEL}")
        
        filename = f"trained_prompts/{obj_conf.DATASET_NAME}_{obj_conf.MODEL}_{obj_conf.VIS_ENCODER.replace('/','')}_opt_{obj_conf.OPTIM_SEED}_spl_{obj_conf.SPLIT_SEED}.pickle"
        if os.path.exists(filename):
            with open(filename, "rb") as f:
                trained_prompts = pickle.load(f)
            learned_prefix = torch.tensor(trained_prompts[0]).to(device)
        else:
            model = VPTBaseline(obj_conf, label_to_idx, device=device, **dict_classes)
            val_accuracy, optimal_prompt = model.train(
                train_seen_dataset, val_seen_dataset, only_seen=True
            )
            save_parameters(optimal_prompt, obj_conf, init_seen=True)
            
            learned_prefix = torch.tensor(optimal_prompt[0]).to(device)

        log.info(f"Let's now train on the unseen classes exploiting learned prompts")
        model = RankVPTBaseline(
            obj_conf, 
            label_to_idx, 
            init_param=learned_prefix,
            device=device, 
            **dict_classes
        )
        val_accuracy, optimal_prompt = model.train(
            train_seen_dataset, val_seen_dataset, train_unseen_dataset
        )
        
        #model.model.prefix = torch.nn.Parameter(torch.tensor(optimal_prompt[0]))

    elif obj_conf.MODEL == "vpt_pseudo_baseline":
        log.info(f"The model in use is: {obj_conf.MODEL}")
        model = VPTPseudoBaseline(obj_conf, label_to_idx, device=device, **dict_classes)
        val_accuracy, optimal_prompt = model.train(
            train_seen_dataset, val_seen_dataset, train_unseen_dataset
        )

    elif obj_conf.MODEL == "quantile_vpt_pseudo_baseline":
        log.info(f"The model in use is: {obj_conf.MODEL}")
        model = QuantileVPTPseudoBaseline(obj_conf, label_to_idx, device=device, **dict_classes)
        val_accuracy, optimal_prompt = model.train(
            train_seen_dataset, val_seen_dataset, train_unseen_dataset
        )

    

    elif obj_conf.MODEL == "quantile_coop_pseudo_baseline":
        log.info(f"The model in use is: {obj_conf.MODEL}")
        model = QuantileCoopPseudoBaseline(obj_conf, label_to_idx, device=device, **dict_classes)
        val_accuracy, optimal_prompt = model.train(
            train_seen_dataset, val_seen_dataset, train_unseen_dataset
        )

    elif obj_conf.MODEL == "all_labeles_vpt_pseudo":
        log.info(f"The model in use is: {obj_conf.MODEL}")
        model = AllVPTPseudoBaseline(obj_conf, label_to_idx, device=device, **dict_classes)
        val_accuracy, optimal_prompt = model.train(
            train_seen_dataset, val_seen_dataset, train_unseen_dataset
        )

    elif obj_conf.MODEL == "all_vpt_pseudo_baseline":
        log.info(f"The model in use is: {obj_conf.MODEL}")
        model = VPTPseudoBaseline(obj_conf, label_to_idx, device=device, **dict_classes)
        val_accuracy, optimal_prompt = model.train(
            train_seen_dataset, val_seen_dataset, train_unseen_dataset
        )

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
    elif obj_conf.MODEL == "iterative_vpt_pseudo":
        log.info(f"The model in use is: {obj_conf.MODEL}")
        model = IterativeFixedPseudo(
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

    elif obj_conf.MODEL == "ablation_teacher_student":
        log.info(f"The model in use is: {obj_conf.MODEL}")
        model = AblationTeacherStudent(
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
    
    if obj_conf.MODEL != 'clip_baseline' and obj_conf.MODEL != 'after_combo_vpt_baseline':
        # Save prompt
        save_parameters(optimal_prompt, obj_conf)
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
    # Define model name
    obj_conf.MODEL = os.environ["MODEL"]
    if obj_conf.MODEL == 'mix_vpt_baseline' \
    or obj_conf.MODEL == 'combo_vpt_baseline' \
    or obj_conf.MODEL == 'after_combo_vpt_baseline' \
    or obj_conf.MODEL == 'rank_vpt_baseline' \
    or obj_conf.MODEL == 'cat_vpt_baseline':
        # Define split seed
        obj_conf.ALPHA = float(os.environ["ALPHA"])
    # Define split seed
    obj_conf.SPLIT_SEED = int(os.environ["SPLIT_SEED"])
    # Define dataset's template for textual prompts
    obj_conf.PROMPT_TEMPLATE = dataset_custom_prompts[obj_conf.DATASET_NAME]
    # Define data dir
    dataset_dir = obj_conf.DATASET_DIR
    # Int prefix
    #obj_conf.PREFIX_SIZE = int(os.environ["PREFIX_SIZE"])
    # Int prefix
    #obj_conf.TYPE = os.environ["TYPE"]
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
