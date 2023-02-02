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
from ..models import CustomTextEncoder, make_scheduler

g = torch.Generator()
g.manual_seed(0)

log = logging.getLogger(__name__)

class TextPrefixModel(nn.Module):
    def __init__(self, initial_prefix, 
                 text_encoder, classes, 
                 temperature=0.07, device='cpu'):
        super(TextPrefixModel, self).__init__()
        self.device = device
        self.initialized_prefix = initial_prefix
        self.classes = classes
        
        self.prefix = nn.Parameter(initial_prefix)
        self.text_encoder = text_encoder

    def forward(self):
        out = self.text_encoder(self.prefix, self.classes)
        norm_out = out / out.norm(dim=-1, keepdim=True)

        return out

class CoopBaseline(object):
    def __init__(self, config, label_to_idx, 
                 classes, seen_classes, unseen_classes,
                 device):

        self.config = config
        
        self.classes = classes
        self.seen_classes = seen_classes
        self.unseen_classes = unseen_classes
        self.label_to_idx = label_to_idx

        seen_to_idx = {c:idx for idx, c in enumerate(self.seen_classes)}
        self.idx_to_real = {seen_to_idx[c]:self.label_to_idx[c] for c in self.seen_classes}
        self.real_to_idx = {self.label_to_idx[c]:seen_to_idx[c] for c in self.seen_classes}


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
        initial_prefix = torch.normal(config.MEAN_INIT, config.VAR_INIT, size=prefix_dim).to(device)

        self.model = TextPrefixModel(initial_prefix,
                                     self.text_encoder,
                                     [' '.join(c.split('_')) for c in self.seen_classes],
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

    def train(self, train_data, val_data):

        # Declare the data pre processing for train and validation data
        train_data.transform = self.transform
        val_data.transform = self.transform

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
                                                                   accum_iter, epoch)
            accelerator.wait_for_everyone()
            if accelerator.is_local_main_process:
                log.info(f"Loss Epoch {epoch}: {total_loss/(len(train_loader))}")
            
            accelerator.free_memory()
            
            if val_loader is not None:
                val_accuracy = self._run_validation(val_loader, epoch)
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    best_prompt = epoch_parameters
            else:
                best_val_accuracy = None
                best_prompt = epoch_parameters
        
        return best_val_accuracy, best_prompt

    def _train_epoch(self, loss, total_loss, train_loader, accum_iter, epoch):

        predictions = []
        images = []
        labels = []
        for i, (img, _, _, label, img_path) in enumerate(tqdm(train_loader)):
            text_features = self.model()
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            with torch.no_grad():
                image_features = self.clip_model.encode_image(img)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # cosine similarity as logits        
            logit_scale = self.clip_model.logit_scale.exp()
            logits = logit_scale * image_features @ text_features.t()

            idx_preds = torch.argmax(logits, dim=1)
            real_preds = [self.seen_classes[i.item()] for i in idx_preds]            
            
            predictions += real_preds
            labels += [self.classes[i.item()] for i in label]
            images += [i for i in img_path]

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

        predictions_outputs = accelerator.gather(predictions)
        labels_outputs = accelerator.gather(labels)
        image_outputs = accelerator.gather(images)

        accuracy = np.sum(np.array(predictions_outputs) == np.array(labels_outputs))/len(predictions_outputs)
        log.info(F"Training accuracy after Epoch {epoch}: {accuracy}")

        current_lr = self.scheduler.get_last_lr()
        self.scheduler.step()

        unwrapped_model = accelerator.unwrap_model(self.model)
        epoch_parameters = [unwrapped_model.prefix.detach().cpu().numpy()]

        return loss, total_loss, epoch_parameters

    def _run_validation(self, val_loader, epoch):
        
        predictions = []
        labels = []
        for img, _, _, label, img_path in tqdm(val_loader):
            text_features = self.model()
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

        predictions_outputs = accelerator.gather(predictions)
        labels_outputs = accelerator.gather(labels)

        accuracy = np.sum(np.array(predictions_outputs) == np.array(labels_outputs))/len(predictions_outputs)
        log.info(f"Validation accuracy after Epoch {epoch}: {accuracy}")

        return accuracy

    def test_predictions(self, data, standard_zsl=False):
        
        # Declare the data pre processing
        data.transform = self.transform
        # Define the data loader
        test_loader = torch.utils.data.DataLoader(data, 
                                                  batch_size=self.config.BATCH_SIZE)

        accelerator.wait_for_everyone()

        self.model, test_loader = accelerator.prepare(self.model, test_loader)

        if standard_zsl:
            self.model.classes =  self.unseen_classes
        else:
            self.model.classes =  self.classes

        # Get prompts
        text_features = self.model()
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

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

        accelerator.wait_for_everyone()

        predictions_outputs = accelerator.gather(predictions)
        image_outputs = accelerator.gather(images)


        df_predictions = pd.DataFrame({'id': image_outputs, 
                                       'class': predictions_outputs})

        return df_predictions

