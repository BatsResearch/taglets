from tqdm import tqdm
import logging
import numpy as np
import pandas as pd

import clip
import torch
from torch import nn
from PIL import Image
from accelerate import Accelerator
accelerator = Accelerator()

from ..utils import seed_worker
from ..models import CustomTextEncoder, make_scheduler, TextPrefixModel

g = torch.Generator()
g.manual_seed(0)

log = logging.getLogger(__name__)


class CoopBaseline(object):
    def __init__(self, config, label_to_idx, 
                 classes, seen_classes, unseen_classes,
                 device, calibration_coefficient=None):
        """ This class define Coop's training and evaluation.

        :param config: dictionaries of prameters in models_config/coop_baseline_config.yml
        :param label_to_idx: dictionary (key, value):(class name, id)
        :param classes: list of class names
        :param seen_classes: list of seen classes' names
        :param unseen_classes: list of unseen classes' names
        :param device: device in use
        :param calibration_coefficient: ?
        """

        self.config = config
        self.classes = classes
        self.seen_classes = seen_classes
        self.unseen_classes = unseen_classes
        self.label_to_idx = label_to_idx
        self.calibration_coefficient = calibration_coefficient

        # Build dictionaries to correctly label model's predictions
        seen_to_idx = {c:idx for idx, c in enumerate(self.seen_classes)}
        self.idx_to_real = {seen_to_idx[c]:self.label_to_idx[c] \
                            for c in self.seen_classes}
        self.real_to_idx = {self.label_to_idx[c]:seen_to_idx[c] \
                            for c in self.seen_classes}

        self.device = device
        self.clip_model, self.transform = clip.load(self.config.VIS_ENCODER, 
                                                    device=self.device)
        self.template = self.config.PROMPT_TEMPLATE

        if torch.cuda.is_available():
            self.text_encoder = CustomTextEncoder(self.clip_model, self.device, 
                                                  torch.float16).to(self.device)
        else:
            self.text_encoder = CustomTextEncoder(self.clip_model, self.device,
                                                  torch.half).to(self.device)

        log.info(f"Freeze text encoder.")
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        
        # Prefix initialization
        prefix_dim = (1, config.PREFIX_SIZE, self.clip_model.token_embedding.embedding_dim)
        self.initial_prefix = torch.normal(self.config.MEAN_INIT, 
                                           self.config.VAR_INIT, 
                                           size=prefix_dim).to(device)

    def initialize_model(self, classes):
        """ This function simply initialized the model to train.
        It defines:
         - optimizer
         - scheduler
         - loss function

        :param classes: list of classes to consider
        """

        self.model = TextPrefixModel(self.initial_prefix,
                                     self.text_encoder,
                                     [' '.join(c.split('_')) for c in classes],
                                     device=self.device).to(self.device)

        for i, parameter in enumerate(self.model.parameters()):
            if parameter.requires_grad:
                log.info(f"Shape of parameters {i}: {parameter.shape}")

        if self.config.OPTIM == 'SGD':
            self.optimizer = torch.optim.SGD(self.model.parameters(), 
                                             lr=self.config.LR, 
                                             weight_decay=self.config.DECAY,
                                             momentum=0.9)

        self.scheduler = make_scheduler(self.optimizer, self.config)
        self.loss_func = torch.nn.CrossEntropyLoss()

    def create_training_dataset(self, train_data, unlabeled_data=None):
        """ This function create the dataset for training. Specifically, it
        merges pseudo-labels for unseen data and labeled data for seen classes.

        :param train_data: Dataset object - training seen classes (defined in zsl_jpl line 323)
        :param unlabeled_data: Dataset object - dataset of unlabeled data for 
                               unseen classes (defined in zsl_jpl line 328)
        """

        return train_data

    def train(self, train_data, val_data, classes, unlabeled_data=None):
        """ This function defines the training of self.model.

        :param train_data: Dataset object - training dataset of labeled data for 
                           seen classes (defined in zsl_jpl line 323)
        :param val_data: Dataset object - validation dataset of labeled data for
                         seen classes (defined in zsl_jpl line 334)
        :param unlabeled_data: Dataset object - dataset of unlabeled data for 
                               unseen classes (defined in zsl_jpl line 328)
        """

        self.initialize_model(classes)
        # Define training dataset
        train_data = self.create_training_dataset(train_data, unlabeled_data)
        
        # Declare the data pre processing for train and validation data
        train_data.transform = self.transform
        val_data.transform = self.transform
        log.info(f"Training data size: {len(train_data.filepaths)}")

        train_loader = torch.utils.data.DataLoader(train_data,
                                                   batch_size=self.config.BATCH_SIZE,
                                                   shuffle=True, worker_init_fn=seed_worker,
                                                   generator=g)
        val_loader = torch.utils.data.DataLoader(val_data,
                                                 batch_size=self.config.BATCH_SIZE)


        accelerator.wait_for_everyone()
        
        self.model, self.optimizer, \
        train_loader, val_loader = accelerator.prepare(self.model, 
                                                       self.optimizer, 
                                                       train_loader,
                                                       val_loader)

        best_val_accuracy = 0
        best_prompt = None
        loss = None
        if val_loader is not None:
            log.info(f"Size of validation dataset: {len(val_data.filepaths)}")
        
        for epoch in range(self.config.EPOCHS):
            log.info(f"Run Epoch {epoch}")
            total_loss = 0
            accum_iter = self.config.ACCUMULATION_ITER  
            
            loss, total_loss, epoch_parameters = self._train_epoch(loss, total_loss, 
                                                                   train_loader, 
                                                                   accum_iter, epoch,
                                                                   unlabeled_data)
            accelerator.wait_for_everyone()
            if accelerator.is_local_main_process:
                log.info(f"Loss Epoch {epoch}: {total_loss/(len(train_loader))}")
            
            accelerator.free_memory()
            
            if val_loader is not None:
                val_accuracy = self._run_validation(val_loader)
                log.info(f"Validation accuracy after Epoch {epoch}: {val_accuracy}")
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    best_prompt = epoch_parameters
            else:
                best_val_accuracy = None
                best_prompt = epoch_parameters

            if unlabeled_data:
                # After validation on seen classes redefine the set of training classes
                self.model.classes = self.classes
        
        return best_val_accuracy, best_prompt

    def _train_epoch(self, loss, total_loss, train_loader, 
                     accum_iter, epoch, unlabeled_data):
        """ This function defines the training epoch of self.model.

        :param loss: float loss (average across batches)
        :param total_loss: float total loss
        :param train_loader: Dataloader object - training data defined in self.train
        :param accum_iter: number of accumulation steps minimum 1
        :param epoch: current epoch
        :param unlabeled_data: boolean. If None running seen supervised coop. 
                               Self-training coop otherwise (CoopPseudoBaseline)
        """

        predictions = []
        labels = []
        for i, (img, _, _, label, img_path) in enumerate(tqdm(train_loader)):
            text_features = self.model(self.model.module.classes)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            with torch.no_grad():
                image_features = self.clip_model.encode_image(img)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # cosine similarity as logits        
            logit_scale = self.clip_model.logit_scale.exp()
            logits = logit_scale * image_features @ text_features.t()

            idx_preds = torch.argmax(logits, dim=1)
            if unlabeled_data:
                real_preds = [self.classes[i.item()] for i in idx_preds]
            else:
                real_preds = [self.seen_classes[i.item()] for i in idx_preds]
            
            predictions += real_preds
            labels += [self.classes[i.item()] for i in label]

            if unlabeled_data:
                labs = torch.tensor([l.item() for l in label]).to(self.device)
            else:
                labs = torch.tensor([self.real_to_idx[l.item()] for l in label]).to(self.device)
            loss = self.loss_func(logits, labs)
            total_loss += loss.item()
            
            accelerator.wait_for_everyone()
            
            loss = loss / accum_iter 
            accelerator.backward(loss)

            # Accumulate grandient
            if ((i + 1) % accum_iter == 0) or (i + 1 == len(train_loader)):
                self.optimizer.step()
                self.model.zero_grad()

        accelerator.wait_for_everyone()

        predictions = torch.tensor([self.label_to_idx[p] for p in predictions]).to(self.device)
        labels = torch.tensor([self.label_to_idx[l] for l in labels]).to(self.device)
        
        predictions_outputs = accelerator.gather(predictions)
        labels_outputs = accelerator.gather(labels)
        
        accuracy = torch.sum(predictions_outputs == labels_outputs)/len(predictions_outputs)
        log.info(F"Training accuracy after Epoch {epoch}: {accuracy}")

        current_lr = self.scheduler.get_last_lr()
        self.scheduler.step()

        unwrapped_model = accelerator.unwrap_model(self.model)
        epoch_parameters = [unwrapped_model.prefix.detach().cpu().numpy()]

        return loss, total_loss, epoch_parameters

    def _run_validation(self, val_loader):
        """ This function computes the validation accuracy on labeled seen data.

        :param val_loder: Dataloader object - validation dataset
        """
        
        predictions = []
        labels = []
        for img, _, _, label, img_path in tqdm(val_loader):
            self.model.classes = self.seen_classes
            text_features = self.model(self.model.classes)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            with torch.no_grad():
                image_features = self.clip_model.encode_image(img)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            logit_scale = self.clip_model.logit_scale.exp()
            logits = logit_scale * image_features @ text_features.t()

            idx_preds = torch.argmax(logits, dim=1)
            real_preds = [self.seen_classes[i.item()] for i in idx_preds]
            
            predictions += real_preds
            labels += [self.classes[i.item()] for i in label]
            

        accelerator.wait_for_everyone()

        predictions = torch.tensor([self.label_to_idx[p] for p in predictions]).to(self.device)
        labels = torch.tensor([self.label_to_idx[l] for l in labels]).to(self.device)
        
        predictions_outputs = accelerator.gather(predictions)
        labels_outputs = accelerator.gather(labels)

        accuracy = torch.sum(predictions_outputs == labels_outputs)/len(predictions_outputs)

        return accuracy

    def test_predictions(self, data, standard_zsl=False):
        """ This function computes predictions on test data.

        :param data: Dataset object - test dataset
        """

        # Declare the data pre processing
        data.transform = self.transform
        # Define the data loader
        test_loader = torch.utils.data.DataLoader(data, 
                                                  batch_size=self.config.BATCH_SIZE)


        self.model, test_loader = accelerator.prepare(self.model, test_loader)

        if standard_zsl:
            self.model.classes =  self.unseen_classes
        else:
            self.model.classes =  self.classes

        accelerator.wait_for_everyone()
   
        # Get prompts
        text_features = self.model(self.model.classes)

        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        log.info(f"TEXT FEATURES SHAPE: {text_features.size()}")

        log.info(f"Start inference for test data")
        predictions = []
        images = []
        for img, _, _, img_path in tqdm(test_loader):
            with torch.no_grad():
                image_features = self.clip_model.encode_image(img)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                # cosine similarity as logits        
            logit_scale = self.clip_model.logit_scale.exp()
            logits = logit_scale * image_features @ text_features.t()
            idx_preds = torch.argmax(logits, dim=1)

            if standard_zsl:
                predictions += [self.unseen_classes[i] for i in idx_preds]
            else:
                predictions += [self.classes[i] for i in idx_preds]

            images += [i for i in img_path]

        predictions = torch.tensor([self.label_to_idx[p] for p in predictions]).to(self.device)
        images = torch.tensor([int(img.split('_')[-1].split('.')[0]) for img in images]).to(self.device)

        accelerator.wait_for_everyone()

        predictions_outputs = accelerator.gather(predictions)
        image_outputs = accelerator.gather(images)

        predictions_outputs = [self.classes[p] for p in predictions_outputs]
        log.info(f"Number of predictions: {len(predictions_outputs)}")
        log.info(f"Number of images: {len(images)}")
        log.info(f"Number of set images: {len(set(images))}")
        log.info(f"Number samples in dataloader: {len(test_loader)}")
        image_outputs = [f"img_{i}.jpg" for i in image_outputs]
        df_predictions = pd.DataFrame({'id': image_outputs, 
                                       'class': predictions_outputs})

        log.info(f"See predictions: {df_predictions.head(5)}")
        log.info(f"See predictions: {df_predictions.shape}")

        return df_predictions

