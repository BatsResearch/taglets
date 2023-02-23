from tqdm import tqdm
import logging
import numpy as np
import pandas as pd
import scipy.stats as st

import clip
import torch
from torch import nn
from PIL import Image
from accelerate import Accelerator
accelerator = Accelerator()

from ..utils import seed_worker
from ..models import CustomImageEncoder, make_scheduler, ImagePrefixModel


g = torch.Generator()
g.manual_seed(0)

log = logging.getLogger(__name__) 

class VPTBaseline(object):
    def __init__(self, config, label_to_idx, 
                 classes, seen_classes, unseen_classes,
                 device):
        """ This class defines Coop's training and evaluation.

        :param config: dictionaries of prameters in models_config/vpt_baseline_config.yml
        :param label_to_idx: dictionary (key, value):(class name, id)
        :param classes: list of class names
        :param seen_classes: list of seen classes' names
        :param unseen_classes: list of unseen classes' names
        :param device: device in use
        """

        self.config = config
        self.classes = classes
        self.seen_classes = seen_classes
        self.unseen_classes = unseen_classes
        self.label_to_idx = label_to_idx

        self.device = device
        self.clip_model, self.transform = clip.load(self.config.VIS_ENCODER, 
                                                    device=self.device)
        self.template = self.config.PROMPT_TEMPLATE
        
        visual_transformer = self.clip_model.visual
        self.image_encoder = CustomImageEncoder(visual_transformer).to(self.device)
        log.info(f"Freeze visual encoder.")
        for param in self.image_encoder.parameters():
            param.requires_grad = False

        self.vis_initial_prefix, self.initial_pos_emb = self.initialize_model_parameters(visual_transformer)

    def define_model(self, teacher=False):
        """ This function allows to define the model and its
        - optimizer
        - schedule
        - loss function """
        # Define model
        self.model = ImagePrefixModel(self.vis_initial_prefix,
                                      self.initial_pos_emb,
                                      self.image_encoder,
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

    def initialize_model_parameters(self, visual_transformer=None):
        """ This function defines the models' parameters initialization.
        """
        
        # Initialize prefix and pos embeddings for visual prompts
        width = visual_transformer.class_embedding.size()[0]
        scale = width ** -0.5
        if self.config.VIS_PREFIX_INIT == 'normal':
            vis_initial_prefix = scale * torch.randn(self.config.PREFIX_SIZE, width)
            if self.config.POS_ENC_INIT == 'same':
                initial_pos_emb = scale * torch.randn(self.config.PREFIX_SIZE, width)
            else:
                initial_pos_emb = torch.zeros(self.config.PREFIX_SIZE, width)
        elif self.config.VIS_PREFIX_INIT == 'uniform':
            val = math.sqrt(6. / float(3 * reduce(mul, (16, 16), 1) + width))  # noqa
            vis_initial_prefix = torch.zeros(self.config.PREFIX_SIZE, width)
            vis_initial_prefix = scale * nn.init.uniform_(vis_initial_prefix, -val, val)
            if self.config.POS_ENC_INIT == 'same':
                initial_pos_emb = torch.zeros(self.config.PREFIX_SIZE, width)
                initial_pos_emb = scale * nn.init.uniform_(initial_pos_emb, -val, val)
            else:
                initial_pos_emb = torch.zeros(self.config.PREFIX_SIZE, width)

        return vis_initial_prefix, initial_pos_emb

    def create_training_dataset(self, train_data, unlabeled_data=None):
        """ This function create the dataset for training. Specifically, it
        merges pseudo-labels for unseen data and labeled data for seen classes.

        :param train_data: Dataset object - training seen classes (defined in zsl_jpl line 323)
        :param unlabeled_data: Dataset object - dataset of unlabeled data for 
                               unseen classes (defined in zsl_jpl line 328)
        """

        return train_data
            
    def train(self, train_data, val_data, unlabeled_data=None, only_unlabelled=False):
        """ This function defines the training of self.model.

        :param train_data: Dataset object - training dataset of labeled data for 
                           seen classes (defined in zsl_jpl line 323)
        :param val_data: Dataset object - validation dataset of labeled data for
                         seen classes (defined in zsl_jpl line 334)
        :param unlabeled_data: Dataset object - dataset of unlabeled data for 
                               unseen classes (defined in zsl_jpl line 328)
        :param only_unlabelled: boolean. It is True if the training only involves 
                                pseudo-labeled unseen data
        """

        self.define_model()
        # Define training dataset
        self.create_training_dataset(train_data, unlabeled_data)

        # Declare the data pre processing for train and validation data
        train_data.transform = self.transform
        val_data.transform = self.transform

        train_loader = torch.utils.data.DataLoader(train_data,
                                                   batch_size=self.config.BATCH_SIZE,
                                                   shuffle=True, worker_init_fn=seed_worker,
                                                   generator=g)
        if self.val_unseen_files is not None:
            seen_imgs = val_data.filepaths
            seen_labs = [self.label_to_idx[l] for l in val_data.labels]

            unseen_imgs = list(self.val_unseen_files)
            unseen_labs = list(self.val_unseen_labs)

            val_data.filepaths = list(unseen_imgs) + list(seen_imgs)
            val_data.labels = list(unseen_labs) + list(seen_labs)
            val_data.label_id = True

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
                                                                   only_unlabelled=only_unlabelled)
            accelerator.wait_for_everyone()
            if accelerator.is_local_main_process:
                log.info(f"Loss Epoch {epoch}: {total_loss/(len(train_loader))}")
            
            accelerator.free_memory()
            
            if val_loader is not None:
                val_accuracy = self._run_validation(val_loader, only_unlabelled)
                log.info(f"Validation accuracy after Epoch {epoch}: {val_accuracy}")
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    best_prompt = epoch_parameters
            else:
                best_val_accuracy = None
                best_prompt = epoch_parameters
        
        return best_val_accuracy, best_prompt

    def define_textual_prompts(self, only_unlabelled=False, validation=False):
        """ This function returns the textual prompts. You can modify the list
        of classes of interest.

        :param only_unlabelled: boolean. It is True if the training only involves 
                                pseudo-labeled unseen data
        """
        return [f"{self.template}{' '.join(i.split('_'))}" \
                            for i in self.seen_classes]

    def reindex_predicted_labels(self, idx_preds, only_unlabelled=False):
        """ This function returns the correct index of predictions to compute
        model's accuracy.

        :param idx_pred: list of predictions ids
        :param only_unlabelled: boolean. It is True if the training only involves 
                                pseudo-labeled unseen data
        """
        return [self.seen_classes[i.item()] for i in idx_preds]
    
    def reindex_true_labels(self, label, only_unlabelled=False):
        """ This function returns the correct index of true labels.

        :param label: list of labels from data loader
        :param only_unlabelled: boolean. It is True if the training only involves 
                                pseudo-labeled unseen data
        """

        return torch.tensor([self.seen_classes.index(self.classes[l.item()]) \
                             for l in label])
    
    def define_loss_function(self, logits, labs, teacher=False):
        
        return self.loss_func(logits, labs)

    def training_model(self, img, teacher=False):
        """ This function allows to customize the model to use while trainig

        :param img: Tensor of images form Dataloader
        """
        return self.model(img)

    def backpropagate(self, teacher=False):

        self.optimizer.step()
        self.model.zero_grad()

    def update_scheduler(self, teacher=False):

        current_lr = self.scheduler.get_last_lr()
        self.scheduler.step()

    def unwrap_model(self, teacher=False):

        return accelerator.unwrap_model(self.model)


    def _train_epoch(self, loss, total_loss, train_loader, 
                     accum_iter, epoch, 
                     only_unlabelled=False,
                     teacher=False):
        """ This function defines the training epoch of self.model.

        :param loss: float loss (average across batches)
        :param total_loss: float total loss
        :param train_loader: Dataloader object - training data defined in self.train
        :param accum_iter: number of accumulation steps minimum 1
        :param epoch: current epoch
        :param only_unlabelled: boolean. It is True if the training only involves 
                                pseudo-labeled unseen data
        :param teachet: boolean. Added to use this function in more subclasses
        """

        # Define text queries
        prompts = self.define_textual_prompts(only_unlabelled)
        log.info(f"Number of prompts: {len(prompts)}")
        
        # Encode text
        with torch.no_grad():
            text = clip.tokenize(prompts).to(self.device)
            text_features = self.clip_model.encode_text(text)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        predictions = []
        labels = []
        for i, (img, _, _, label, img_path) in enumerate(tqdm(train_loader)):
            image_features = self.training_model(img, teacher)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # cosine similarity as logits        
            logit_scale = self.clip_model.logit_scale.exp()
            logits = logit_scale * image_features @ text_features.t()
            idx_preds = torch.argmax(logits, dim=1)
            
            real_preds = self.reindex_predicted_labels(idx_preds, only_unlabelled)
                
            predictions += real_preds
            labels += [self.classes[i.item()] for i in label]

            labs = self.reindex_true_labels(label, only_unlabelled)
            idx_seen = [self.classes.index(c) for c in self.seen_classes]
            idx_unseen = [self.classes.index(c) for c in self.unseen_classes]

            count_seen = len([l for l in labs if l in idx_seen])
            count_unseen = len([l for l in labs if l in idx_unseen])
            self.balance_param = count_seen/count_unseen

            labs = labs.to(self.device)
            loss = self.define_loss_function(logits, labs, teacher)
            total_loss += loss.item()
            
            accelerator.wait_for_everyone()
            
            loss = loss / accum_iter 
            accelerator.backward(loss)

            # Accumulate grandient
            if ((i + 1) % accum_iter == 0) or (i + 1 == len(train_loader)):
                self.backpropagate(teacher)

        accelerator.wait_for_everyone()

        predictions = torch.tensor([self.label_to_idx[p] for p in predictions]).to(self.device)
        labels = torch.tensor([self.label_to_idx[l] for l in labels]).to(self.device)

        predictions_outputs = accelerator.gather(predictions)
        labels_outputs = accelerator.gather(labels)

        # Get harmonic mean
        idx_seen = [self.label_to_idx[c] for c in self.seen_classes] 
        seen_true = [i for i, c in enumerate(labels_outputs) if c in idx_seen]
        seen_preds = predictions_outputs[seen_true]
        seen_labs = labels_outputs[seen_true]
        seen_accuracy = torch.sum(seen_preds == seen_labs)/len(seen_true)
        
        idx_unseen = [self.label_to_idx[c] for c in self.unseen_classes] 
        unseen_true = [i for i, c in enumerate(labels_outputs) if c in idx_unseen]
        unseen_preds = predictions_outputs[unseen_true]
        unseen_labs = labels_outputs[unseen_true]
        unseen_accuracy = torch.sum(unseen_preds == unseen_labs)/len(unseen_true)
        
        accuracy = st.hmean([unseen_accuracy.cpu(), seen_accuracy.cpu()])

        #accuracy = torch.sum(predictions_outputs == labels_outputs)/len(predictions_outputs)
        log.info(F"Training SEEN accuracy after Epoch {epoch}: {seen_accuracy}")
        log.info(F"Training UNSEEN accuracy after Epoch {epoch}: {unseen_accuracy}")
        log.info(F"Training HARMONIC accuracy after Epoch {epoch}: {accuracy}")

        self.update_scheduler(teacher)

        unwrapped_model = self.unwrap_model(teacher)
        epoch_parameters = [unwrapped_model.prefix.detach().cpu().numpy(), 
                            unwrapped_model.image_pos_emb.detach().cpu().numpy()]

        return loss, total_loss, epoch_parameters

    def _run_validation(self, val_loader, only_unlabelled=False, teacher=False):
        """ This function computes the validation accuracy on labeled seen data.

        :param val_loder: Dataloader object - validation dataset
        """

        # Define text queries
        if self.val_unseen_files is not None:
            val = False
        else:
            val = True

        prompts = self.define_textual_prompts(only_unlabelled, validation=val)
        log.info(f"Number of prompts: {len(prompts)}")
        
        # Encode text
        with torch.no_grad():
            text = clip.tokenize(prompts).to(self.device)
            text_features = self.clip_model.encode_text(text)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        predictions = []
        labels = []
        for img, _, _, label, img_path in tqdm(val_loader):
            image_features = self.training_model(img, teacher)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            logit_scale = self.clip_model.logit_scale.exp()
            logits = logit_scale * image_features @ text_features.t()

            idx_preds = torch.argmax(logits, dim=1)
            if self.val_unseen_files is not None:
                real_preds = [self.classes[i.item()] for i in idx_preds]
            else:
                real_preds = [self.seen_classes[i.item()] for i in idx_preds]
            
            predictions += real_preds
            labels += [self.classes[i.item()] for i in label]
            

        accelerator.wait_for_everyone()

        predictions = torch.tensor([self.label_to_idx[p] for p in predictions]).to(self.device)
        labels = torch.tensor([self.label_to_idx[l] for l in labels]).to(self.device)

        predictions_outputs = accelerator.gather(predictions)
        labels_outputs = accelerator.gather(labels)

        if len(prompts) < len(self.classes):
            accuracy = torch.sum(predictions_outputs == labels_outputs)/len(predictions_outputs)
        else:
            # Get harmonic mean
            idx_seen = [self.label_to_idx[c] for c in self.seen_classes] 
            seen_true = [i for i, c in enumerate(labels_outputs) if c in idx_seen]
            seen_preds = predictions_outputs[seen_true]
            seen_labs = labels_outputs[seen_true]
            seen_accuracy = torch.sum(seen_preds == seen_labs)/len(seen_true)
            
            idx_unseen = [self.label_to_idx[c] for c in self.unseen_classes] 
            unseen_true = [i for i, c in enumerate(labels_outputs) if c in idx_unseen]
            unseen_preds = predictions_outputs[unseen_true]
            unseen_labs = labels_outputs[unseen_true]
            unseen_accuracy = torch.sum(unseen_preds == unseen_labs)/len(unseen_true)
            
            accuracy = st.hmean([unseen_accuracy.cpu(), seen_accuracy.cpu()])

            log.info(F"Validation SEEN accuracy after Epoch: {seen_accuracy}")
            log.info(F"Validation UNSEEN accuracy after Epoch: {unseen_accuracy}")
            log.info(F"Validation HARMONIC accuracy after Epoch: {accuracy}")

        
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

        accelerator.wait_for_everyone()

        self.model, test_loader = accelerator.prepare(self.model, test_loader)

        # Define text queries
        if standard_zsl:
            prompts = [f"{self.template}{' '.join(i.split('_'))}" \
                        for i in self.unseen_classes]
        else:
            prompts = [f"{self.template}{' '.join(i.split('_'))}" \
                        for i in self.classes]
        log.info(f"Number of prompts: {len(prompts)}")
        # This is required for distributed training
        test_files = [f.split('/')[-1] for f in test_loader.dataset.filepaths]

        # Encode text
        text = clip.tokenize(prompts).to(self.device)
        text_features = self.clip_model.encode_text(text)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        log.info(f"Start inference for test data")
        predictions = []
        images = []
        for img, _, _, img_path in tqdm(test_loader):
            with torch.no_grad():
                image_features = self.model(img)
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
        images = torch.tensor([test_files.index(img) for img in images]).to(self.device)
        
        accelerator.wait_for_everyone()

        predictions_outputs = accelerator.gather(predictions)
        image_outputs = accelerator.gather(images)

        predictions_outputs = [self.classes[p] for p in predictions_outputs]
        image_outputs = [test_files[i] for i in image_outputs]
        
        df_predictions = pd.DataFrame({'id': image_outputs, 
                                       'class': predictions_outputs})
        df_predictions.drop_duplicates(subset=['id', 'class'], inplace=True)

        return df_predictions