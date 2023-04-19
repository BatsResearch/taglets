import json
import logging
import os
import pickle

import numpy as np
import pandas as pd
import scipy.stats as st
import torch
from accelerate import Accelerator

accelerator = Accelerator()


log = logging.getLogger(__name__)


def evaluate_predictions(
    config,
    df_predictions,
    test_labeled_files,
    labels,
    unseen_classes,
    seen_classes=None,
    standard_zsl=False,
):
    df_test = pd.DataFrame({"id": test_labeled_files, "true": labels})
    df_test["id"] = df_test["id"].apply(lambda x: x.split("/")[-1])
    # log.info(f"DF TEST: {df_test.head(5)}")
    # log.info(f"DF PREDS: {df_predictions.head(5)}")
    df_predictions = pd.merge(df_predictions, df_test, on="id")

    if config.LEARNING_PARADIGM == 'ul' or config.LEARNING_PARADIGM == 'ssl':
        accuracy = (
            np.sum(df_predictions["class"] == df_predictions["true"])
            / df_predictions.shape[0]
        )

        return accuracy

    if standard_zsl:
        # df_predictions['true'] = labels
        df_predictions = df_predictions[df_predictions["true"].isin(unseen_classes)]
        accuracy = (
            np.sum(df_predictions["class"] == df_predictions["true"])
            / df_predictions.shape[0]
        )

        return accuracy

    else:
        # df_predictions['true'] = labels
        unseen_predictions = df_predictions[df_predictions["true"].isin(unseen_classes)]
        unseen_accuracy = (
            np.sum(unseen_predictions["class"] == unseen_predictions["true"])
            / unseen_predictions.shape[0]
        )

        seen_predictions = df_predictions[df_predictions["true"].isin(seen_classes)]
        seen_accuracy = (
            np.sum(seen_predictions["class"] == seen_predictions["true"])
            / seen_predictions.shape[0]
        )

        harmonic_mean = st.hmean([unseen_accuracy, seen_accuracy])

        return unseen_accuracy, seen_accuracy, harmonic_mean

def store_results(
    obj_conf, 
    std_response, 
    unseen_accuracy=None, 
    seen_accuracy=None, 
    harmonic_mean=None
):
    """The function stores results of the model in a json.

    :param obj_config: class object that stores configurations

    """
    if obj_conf.LEARNING_PARADIGM == 'trzsl':
        # Store results
        if accelerator.is_local_main_process:
            results_to_store = {
            "model": obj_conf.MODEL,
            "config": obj_conf.__dict__,
            "std_accuracy": std_response,
            "gen_accuracy": harmonic_mean,
            "gen_seen": seen_accuracy,
            "gen_unseen": unseen_accuracy,
        }
    else:
        # Store results
        if accelerator.is_local_main_process:
            results_to_store = {
                "model": obj_conf.MODEL,
                "config": obj_conf.__dict__,
                "std_accuracy": std_response,
            }


    if accelerator.is_local_main_process:
        file_name = f"results_model_{obj_conf.MODEL}.json"

        # Check if the file already exists
        if os.path.exists(file_name):
            # If the file exists, open it in append mode
            with open(file_name, "a") as f:
                # Append the res dictionary to the file
                f.write(json.dumps(results_to_store) + "\n")
        else:
            # If the file doesn't exist, create a new file
            with open(file_name, "w") as f:
                # Write the res dictionary to the file
                f.write(json.dumps(results_to_store) + "\n")

def save_parameters(obj, config, iteration=None):
    """ Save in a pickle the parameters used for 
    evaluation.

    :param obj: object to save
    :param config: object with method configurations
    """

    if iteration is None:
        file_name = f"trained_prompts/{config.DATASET_NAME}_{config.LEARNING_PARADIGM}_{config.MODEL}_{config.VIS_ENCODER.replace('/','')}_opt_{config.OPTIM_SEED}_spl_{config.SPLIT_SEED}.pickle"
    else:
        file_name = f"trained_prompts/{config.DATASET_NAME}_{config.LEARNING_PARADIGM}_{config.MODEL}_{config.VIS_ENCODER.replace('/','')}_iter_{iteration}_opt_{config.OPTIM_SEED}_spl_{config.SPLIT_SEED}.pickle"
    
    if config.MODALITY == 'multi':
        torch.save(obj, file_name)
    else:
        with open(file_name, 'wb') as f:
            pickle.dump(obj, f)


def save_pseudo_labels(imgs, labs, config, iteration):

    filename = f"pseudolabels/{config.DATASET_NAME}_{config.LEARNING_PARADIGM}_{config.MODEL}_{config.VIS_ENCODER.replace('/', '')}_iter_{iteration}_opt_{config.OPTIM_SEED}_spl_{config.SPLIT_SEED}.pickle"
    with open(filename, "wb") as f:
        pickle.dump({"filepaths": imgs, "labels": labs}, f)

