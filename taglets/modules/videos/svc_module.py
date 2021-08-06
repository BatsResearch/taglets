import os
import math
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

    def predict(self, data):
        log.info('Beginning prediction')
        pred_classifier = self.classifier
        
        X = self._extract_features(data, train=False)
        nembs_valid = normalize(X, axis=1)

        preds = pred_classifier.predict(nembs_valid)
        
        return preds

    def _extract_features(self, data, train=True):

        num_proc = 1
        
        if train:
            data_loader = self._get_dataloader(data, shuffle=True)
        else:
            data_loader = self._get_dataloader(data, False)
        
        accelerator.wait_for_everyone()

        #self.model = accelerator.prepare(self.model)
        self.model.to('cuda')
        self.model.eval()

        if train:
            # Init matrices
            # Set matrices dimension
            num_epochs = int(math.ceil(len(data.filepaths) / (self.batch_size * num_proc)))
            len_acc_total = (self.batch_size*num_proc) * num_epochs
            len_acc_total = len(data.filepaths)
            
            #log.info(f"Lengthof eval data:{len_acc_total} and num_epochs: {num_epochs}")
            
            X = np.zeros((len_acc_total, 2304))
            Y = np.zeros((len_acc_total))

            # Batch size dimension
            dim = 0
            
            for batch in data_loader:
                inputs = batch[0]['video']
                labels = batch[1]

                #with torch.set_grad_enabled(False):
                output = self.model(inputs)
                
                output  = accelerator.gather(output.detach())
                labels = accelerator.gather(labels)
                
                #log.info(f'train output: {output.size()}')
                #log.info(f'train labels: {len(labels)}')
                
                # Change indices
                #log.info(f'self.batch_size * num_proc: {self.batch_size * num_proc}')
                max_index = dim + self.batch_size * num_proc
                log.info(f'max_index: {dim} and {max_index}') 
                X[dim:max_index, :] = output.cpu().numpy()[0]
                Y[dim:max_index] = labels.cpu()
                dim += self.batch_size * num_proc # number of processes
                #log.info(f'train dim: {dim}')

            dataset_len = len(data_loader.dataset)
            X = X[:dataset_len, :]
            Y = Y[:dataset_len]
            return X, Y

        else:
            # Init matrices
            num_epochs = int(math.ceil(len(data.filepaths) / (self.batch_size * num_proc)))
            len_acc_total = (self.batch_size*num_proc) * num_epochs
            len_acc_total = len(data.filepaths)
            #log.info(f"Lengthof eval data:{len_acc_total} and num_epochs: {num_epochs}")
            #log.info(f"batch size:{self.batch_size}")
            
            X = np.zeros((len_acc_total, 2304))

            dim = 0
            for batch in data_loader:
                inputs = batch['video']

                #with torch.set_grad_enabled(False):
                output = self.model(inputs)
                output  = accelerator.gather(output.detach())
                #log.info(f'eval output: {output.size()}')
                #log.info(f'self.batch_size * num_proc: {self.batch_size * num_proc}')
                max_index = dim + self.batch_size * num_proc
                log.info(f'max_index: {dim} and {max_index}')  
                X[dim:max_index, :] = output.cpu().numpy()[0]
                
                dim += self.batch_size * num_proc # number of processes
                #log.info(f'eval dim: {dim}')    

            dataset_len = len(data_loader.dataset)
            X = X[:dataset_len, :]

            return X   
