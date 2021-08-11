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
from ...pipeline import VideoTaglet



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
        preds, labels = self.predict(labeled_data)
        correct = (preds == labels).sum()
        
        return correct / len(preds)
        

    def predict(self, data):
        log.info('Beginning prediction')
        pred_classifier = self.classifier
        
        X, Y = self._extract_features(data, train=False)
        nembs_valid = normalize(X, axis=1)

        preds = pred_classifier.predict(nembs_valid)

        if len(Y) > 0:
            return preds, Y
        else:
            return preds

    def _extract_features(self, data, train=True):

        #num_proc = 1
        
        if train:
            data_loader = self._get_dataloader(data, shuffle=True)
        else:
            data_loader = self._get_dataloader(data, shuffle=False)
        
        accelerator.wait_for_everyone()
        self.model.eval()

        if train:
            # Init matrices            
            X = np.array([]).reshape(0, 2304)
            Y = np.array([])
            
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
            # Check if embeddings already computed
            # if os.path.isfile("X.p"):
            #     X = pickle.load(open("X.p", "rb" ))
            #     Y = pickle.load(open("Y.p", "rb" ))

            # else: 
            X = np.array([]).reshape(0, 2304)
            Y = np.array([])

            for batch in data_loader:
                if isinstance(batch, list):
                    inputs = batch[0]['video'] 
                    targets = batch[1]
                else:
                    #if os.path.isfile("X.p"):
                    #    X = pickle.load(open("X.p", "rb" ))
                    #    Y = pickle.load(open("Y.p", "rb" ))
                    #    return X, Y
                    inputs, targets = batch['video'], None

                output = self.model(inputs)
                output  = accelerator.gather(output.detach())
                if targets is not None:
                    labels = accelerator.gather(targets)
                    lab = labels.cpu()
                    Y = np.concatenate((Y, lab), axis=0)
                # Embeddings to cpu
                emb = output.cpu().numpy()
                X = np.concatenate((X, emb), axis=0)


            dataset_len = len(data_loader.dataset)
            X = X[:dataset_len, :]
            Y = Y[:dataset_len]
            
            #if isinstance(batch, list):
            #    return X, Y
            #else:
            #    pickle.dump(X, open("X.p","wb"))
            #    pickle.dump(Y, open("Y.p","wb"))
            return X, Y
