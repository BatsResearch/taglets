import os
import math
import pickle
import logging

import numpy as np
import torch
import torch.nn as nn
from sklearn.svm import LinearSVC
from sklearn.linear_model import (LogisticRegression, 
                                LogisticRegressionCV)
from sklearn.preprocessing import normalize
from accelerate import Accelerator
accelerator = Accelerator()

from ..module import Module
from ...pipeline import VideoTaglet, Cache



log = logging.getLogger(__name__)


class SvcVideoModule(Module):
    """
    A module that fine-tunes the task's initial model.
    """
    def __init__(self, task):
        super().__init__(task)
        self.taglets = [SVCVideoTaglet(task)]


class SVCVideoTaglet(VideoTaglet):
    def __init__(self, task):
        super().__init__(task)
        self.name = 'svc-video'
        #for param in self.model.parameters():
        #    param.requires_grad = False
        
        self.model.to('cuda')
        self.model.blocks[6].proj = nn.Sequential()
        self.classifier = LinearSVC()

        if os.getenv("LWLL_TA1_PROB_TASK") is not None:
            self.save_dir = os.path.join('/home/tagletuser/trained_models', self.name)
        else:
            self.save_dir = os.path.join('trained_models', self.name)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def train(self, train_data, val_data, unlabeled_data=None):
        
        X, Y = self._extract_features(train_data)
        # Training data
        nembs_train = normalize(X, axis=1)
        labs_train = Y
        
        log.info('Beginning training')
        self.classifier = self.classifier.fit(nembs_train, labs_train)

    def evaluate(self, labeled_data):
        preds, labels = self.predict(labeled_data, evaluation=True)
        correct = (preds == labels).sum()
        
        return correct / len(preds)
        

    def predict(self, data, evaluation=False, test=False):
        log.info('Beginning prediction')
        pred_classifier = self.classifier
        
        X, Y = self._extract_features(data, evaluation, test, train=False)
        nembs_valid = normalize(X, axis=1)

        preds = pred_classifier.predict(nembs_valid)

        if len(Y) > 0:
            return preds, Y
        else:
            return preds

    def _extract_features(self, data, evaluation=False, test=False, train=True):

        #num_proc = 1
        
        if train:
            data_loader = self._get_dataloader(data, shuffle=True)
        else:
            data_loader = self._get_dataloader(data, shuffle=False)
        
        accelerator.wait_for_everyone()
        self.model.eval()
        
        # Init matrices            
        X = np.array([]).reshape(0, 2304)
        Y = np.array([])
        
        if train:
            for batch in data_loader:
                inputs = batch[0]['video']
                labels = batch[1]

                output = self.model(inputs)
                output  = accelerator.gather(output.detach())
                labels = accelerator.gather(labels)
                
                # Embeddings to cpu
                emb = output.cpu().numpy()
                X = np.concatenate((X, emb), axis=0)
                # Labels on cpu
                lab = labels.cpu()
                Y = np.concatenate((Y, lab), axis=0)

            dataset_len = len(data_loader.dataset)
            X = X[:dataset_len, :]
            Y = Y[:dataset_len]
            return X, Y

        else:
            if evaluation:
                for batch in data_loader:
                    inputs = batch[0]['video'] 
                    targets = batch[1]

                    output = self.model(inputs)
                    output  = accelerator.gather(output.detach())
                    
                    labels = accelerator.gather(targets)
                    lab = labels.cpu()
                    Y = np.concatenate((Y, lab), axis=0)
                    # Embeddings to cpu
                    emb = output.cpu().numpy()
                    X = np.concatenate((X, emb), axis=0)

                dataset_len = len(data_loader.dataset)
                X = X[:dataset_len, :]
                Y = Y[:dataset_len]
                return X, Y
            elif test == True:
                for batch in data_loader:
                    inputs, targets = batch['video'], None
                    output = self.model(inputs)
                    output  = accelerator.gather(output.detach())
                    # Embeddings to cpu
                    emb = output.cpu().numpy()
                    X = np.concatenate((X, emb), axis=0)
                dataset_len = len(data_loader.dataset)
                X = X[:dataset_len, :]
                return X, Y
            else:
                # Check if embeddings already computed
                eval_embeddings = Cache.get("svc-eval-embeddings", 'other')
                if eval_embeddings is None:
                    for batch in data_loader:
                        inputs, targets = batch['video'], None
                        output = self.model(inputs)
                        output  = accelerator.gather(output.detach())
                        # Embeddings to cpu
                        emb = output.cpu().numpy()
                        X = np.concatenate((X, emb), axis=0)

                    dataset_len = len(data_loader.dataset)
                    X = X[:dataset_len, :]
                    Cache.set('svc-eval-embeddings', X, Y)
                    return X, Y
                else:
                    return eval_embeddings[0], eval_embeddings[1]

