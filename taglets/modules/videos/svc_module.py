import os
import logging

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
        for param in self.model.parameters():
            param.requires_grad = False
        
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
        
        #log.info(f"Data dataLoader {data.filepaths}")
        #data_loader = self._get_dataloader(data, False)
        #dataset_len = len(data_loader.dataset)
        
        #accelerator.wait_for_everyone()
        #self.model = accelerator.prepare(self.model)
        
        X = self._extract_features(data)
        nembs_valid = normalize(X, axis=1)

        preds = pred_classifier.predict(nembs_valid)
        
        return preds

    def _extract_features(self, data, train=True):

        if train:
            data_loader = self._get_dataloader(data, shuffle=True)
        else:
            data_loader = self._get_dataloader(data, False)
        
        accelerator.wait_for_everyone()

        self.model = accelerator.prepare(self.model)
        self.model.eval()

        if train:
            # Init matrices
            X = np.zeros((len(data.filepaths), 2304))
            Y = np.zeros((len(data.filepaths)))

            # Batch size dimension
            dim = 0
            
            for batch in data_loader:
                inputs = batch[0]['video']
                labels = batch[1]

                with torch.set_grad_enabled(False):
                    output = self.model(inputs)
                
                output  = accelerator.gather(output.detach().numpy()[0])
                labels = accelerator.gather(labels.item())
                
                log.info(f'train output: {output}')
                log.info(f'train labels: {labels}')
                # Change indices
                X[dim + self.batch_size,:] = output
                Y[dim + self.batch_size] = labels
                dim += self.batch_size * 8 # number of processes
                log.info(f'train dim: {dim}')

            return X, Y

        else:
            # Init matrices
            X = np.zeros((len(data.filepaths), 2304))

            dim = 0
            for batch in data_loader:
                inputs = batch['video']

                with torch.set_grad_enabled(False):
                    output = self.model(inputs)
                output  = accelerator.gather(output.detach().numpy()[0])
                log.info(f'eval output: {output}')
                
                X[dim + self.batch_size,:] = output
                dim += self.batch_size * 8 # number of processes
                log.info(f'eval dim: {dim}')     

            return X   
