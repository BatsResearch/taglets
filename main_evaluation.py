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
from methods.semi_supervised_learning import (
    MultimodalPrompt,
    TextualPrompt,
    VisualPrompt,
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
    test_labeled_files, test_labeles = zip(*test_data)
    label_to_idx = {c: idx for idx, c in enumerate(classes)}

    DatasetObject = dataset_object(obj_conf.DATASET_NAME)
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
    log.info(f"Len test data: {len(test_dataset.filepaths)}")
    log.info(f"\n-------------------------------------------------------------\n")
    # Define model
    device = "cuda" if torch.cuda.is_available() else "cpu"

    folder = 'trained_prompts'
    # params = 'RESICS45_ssl_textual_fpl_ViT-B32_opt_1_spl_500.pickle'
    params = 'RESICS45_ssl_visual_fpl_ViT-B32_opt_1_spl_500.pickle'
    file_param = f'{folder}/{params}'
    with open(file_param, 'rb') as f:
        prompts = pickle.load(f)

    # params = [
    #     'RESICS45_ul_multimodal_fpl_ViT-B32_opt_1_spl_500_coop_embeddings.pickle',
    #     'RESICS45_ul_multimodal_fpl_ViT-B32_opt_1_spl_500_deep_vpt.pickle',
    #     'RESICS45_ul_multimodal_fpl_ViT-B32_opt_1_spl_500_proj_coop_post.pt',
    #     'RESICS45_ul_multimodal_fpl_ViT-B32_opt_1_spl_500_proj_coop_pre.pt',
    #     'RESICS45_ul_multimodal_fpl_ViT-B32_opt_1_spl_500_proj_vpt_post.pt',
    #     'RESICS45_ul_multimodal_fpl_ViT-B32_opt_1_spl_500_proj_vpt_pre.pt',
    #     'RESICS45_ul_multimodal_fpl_ViT-B32_opt_1_spl_500_transformer.pt',
    #     'RESICS45_ul_multimodal_fpl_ViT-B32_opt_1_spl_500_vpt_embeddings.pickle',
    # ]
    # list_dicts = [
    #     'transformer', 
    #     'proj_coop_pre',
    #     'proj_coop_post',
    #     'proj_vpt_pre',
    #     'proj_vpt_post',
    # ]

    # dict_param = {}
    # for p in params:
    #     file_param = f'{folder}/{p}'
    #     name = '_'.join(p.split('_')[9:]).split('.')[0]
    #     if name in list_dicts:
    #         dict_param[name] = torch.load(file_param, map_location=torch.device('cpu'))
    #     else:
    #         print(name, file_param)
    #         with open(file_param, 'rb') as f:
    #             dict_param[name] = pickle.load(f)
    
    # log.info(f"Len params: {len(dict_param)}")

    
    
    log.info(f"\n----------------------MODEL INFO-----------------------\n")
    if obj_conf.MODALITY == "text":
        model = TextualPrompt(
            obj_conf, 
            label_to_idx, 
            device=device, 
            **dict_classes
        )
        model.load_model_eval()

        model.model.prefix = torch.nn.Parameter(torch.tensor(prompts[0]))

    elif obj_conf.MODALITY == "image":
        model = VisualPrompt(
            obj_conf, 
            label_to_idx, 
            device=device, 
            **dict_classes
        )
        model.load_model_eval()

        model.model.prefix = torch.nn.Parameter(torch.tensor(prompts[0]))

    elif obj_conf.MODALITY == "multi":
        model = MultimodalPrompt(
            obj_conf, 
            label_to_idx, 
            device=device, 
            **dict_classes
        )
        model.load_model_eval()

        model.model.coop_embeddings = torch.nn.Parameter(torch.tensor(dict_param['coop_embeddings']))
        model.model.vpt_embeddings = torch.nn.Parameter(torch.tensor(dict_param['vpt_embeddings']))
        
        model.model.proj_coop_pre.load_state_dict(dict_param['proj_coop_pre'])
        model.model.proj_coop_post.load_state_dict(dict_param['proj_coop_post'])
        
        model.model.proj_vpt_pre.load_state_dict(dict_param['proj_vpt_pre'])
        model.model.proj_vpt_post.load_state_dict(dict_param['proj_vpt_post'])

        model.model.transformer.load_state_dict(dict_param['transformer'])
        

    # Validate on test set (standard)
    images, predictions, prob_preds = model.evaluation(test_dataset)

    log.info(f"IMAGES: {np.array(images[:3])}")
    log.info(f"PREDICTIONS: {predictions[:3]}")
    log.info(f"LABELS: {test_labeles[:3]}")
    log.info(f"PROBS: {prob_preds[0]}")

    dictionary_predictions = {
        'images' : images, 
        'predictions' : predictions,
        'labels' : test_labeles,
        'logits' : prob_preds,
    }

    with open(f'evaluation/{params}', 'wb') as f:
        pickle.dump(dictionary_predictions, f)

    # Store model results
    # store_results(obj_conf, std_response)
    sys.exit()
 
def main():
    parser = argparse.ArgumentParser(description="Run JPL task")
    parser.add_argument(
        "--model_config",
        type=str,
        default="model_config.yml",
        help="Name of model config file",
    )
    parser.add_argument(
        "--learning_paradigm",
        type=str,
        default="trzsl",
        help="Choose among trzsl, ssl, and ul",
    )
    # dataset
    # modality
    # model
    # encoder
    # optim seed
    # split

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
    # Define split seed
    obj_conf.SPLIT_SEED = int(os.environ["SPLIT_SEED"])
    # Define dataset's template for textual prompts
    obj_conf.PROMPT_TEMPLATE = dataset_custom_prompts[obj_conf.DATASET_NAME]
    # Define data dir
    dataset_dir = obj_conf.DATASET_DIR
    # Int prefix
    obj_conf.LEARNING_PARADIGM = args.learning_paradigm
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
