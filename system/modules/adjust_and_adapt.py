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
from pytorch_metric_learning import losses 

from ..data import CustomDataset
from ..utils import seed_worker, composed_transform, \
                    prepare_data_ssl_loss
from ..models import CustomImageEncoder, make_scheduler

g = torch.Generator()
g.manual_seed(0)

log = logging.getLogger(__name__)

class ImagePrefixModel(nn.Module):
    def __init__(self, initial_prefix, 
                 initial_pos_emb, image_encoder, 
                 temperature=0.07, device='cpu'):
        super(ImagePrefixModel, self).__init__()
        self.device = device
        self.initialized_prefix = initial_prefix
        
        # Initialize the model's parametwets
        self.prefix = nn.Parameter(initial_prefix) 
        self.image_pos_emb = nn.Parameter(initial_pos_emb) 
        self.image_encoder = image_encoder

    def forward(self, x):
        # Combine prefix and class embeddings to get the entire prompt representation for the
        # two augmented images
        out = self.image_encoder(x, self.prefix, self.image_pos_emb)
        norm_out = out / out.norm(dim=-1, keepdim=True)

        return out

class Image2TextAdapter(nn.Module):
    def __init__(self, image_encoder, 
                 classes,
                 n_layers=3, 
                 device='cpu'):
        super(Image2TextAdapter, self).__init__()
        self.device = device
        self.image_encoder = image_encoder

        self.n_layers = n_layers
        
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.linears = nn.ModuleList([nn.Linear(in_feat, out_feat) \
                                      for i in range(self.n_layers)])

    def forward(self, x):
        # Combine prefix and class embeddings to get the entire prompt representation for the
        # two augmented images
        out = self.image_encoder(x, self.prefix, self.image_pos_emb)
        norm_out = out / out.norm(dim=-1, keepdim=True)

        return out

class AdjustAndAdapt(object):
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
        
        visual_transformer = self.clip_model.visual
        self.image_encoder = CustomImageEncoder(visual_transformer).to(self.device)
        log.info(f"Freeze visual encoder.")
        for param in self.image_encoder.parameters():
            param.requires_grad = False

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

        # Define model
        self.vpt_model = ImagePrefixModel(vis_initial_prefix,
                                      initial_pos_emb,
                                      self.image_encoder,
                                      device=self.device).to(self.device)

        for i, parameter in enumerate(self.vpt_model.parameters()):
            if parameter.requires_grad:
                log.info(f"Shape of parameters {i}: {parameter.shape}")

        if self.config.OPTIM == 'SGD':
            self.vpt_optimizer = torch.optim.SGD(self.vpt_model.parameters(), 
                                                 lr=self.config.LR, 
                                                 weight_decay=self.config.DECAY,
                                                 momentum=0.9)

        self.vpt_scheduler = make_scheduler(self.vpt_optimizer, self.config)
        self.ssl_loss = losses.NTXentLoss(temperature=self.config.TEMPERATURE)

    def get_clip_pseudo_labels(self):
        pass

    def train_visual_prompts(self):
        
        
        # Define data augmentations
        transform_base = self.config.AUGMENTATION_BASE
        transform_strong = self.config.AUGMENTATION_STRONG
        
        augs = (composed_transform('base', transform_base=transform_base, preprocess_transform=self.transform),
                composed_transform('strong', transform_strong=transform_strong, preprocess_transform=self.transform))
        # Define dataset to use for training
        training_unlabeled_data = CustomDataset(self.unsupervised_training_data, 
                                                self.data_folder, transform=self.transform,
                                                augmentations=augs, train=True, labels=None,
                                                label_map=self.label_to_idx)
        log.info(f"Unsupervised training set is of length {len(training_unlabeled_data.filepaths)}")
        
        unsupervised_train_loader = torch.utils.data.DataLoader(training_unlabeled_data,
                                                                batch_size=self.config.BATCH_SIZE, 
                                                                shuffle=True, worker_init_fn=seed_worker, 
                                                                generator=g)
        
        accelerator.wait_for_everyone()
        self.vpt_model, self.vpt_optimizer, \
        train_loader = accelerator.prepare(self.vpt_model,
                                           self.vpt_optimizer, 
                                           unsupervised_train_loader)
        loss =  None
        for epoch in range(self.config.EPOCHS):
            log.info(f"Run Epoch {epoch}")
            total_loss = 0
            accum_iter = self.config.ACCUMULATION_ITER  
            loss, total_loss, epoch_parameters = self._train_unsupervised_epoch(loss, total_loss, 
                                                                                unsupervised_train_loader, 
                                                                                accum_iter)
            accelerator.wait_for_everyone()
            if accelerator.is_local_main_process:
                log.info(f"Loss Epoch {epoch}: {total_loss/(len(unsupervised_train_loader))}")
            
        accelerator.free_memory()

        return epoch_parameters

    def _train_unsupervised_epoch(self, loss, total_loss, 
                                  unsupervised_train_loader, 
                                  accum_iter):
        for i, (img, aug_1, aug_2, img_path) in enumerate(tqdm(unsupervised_train_loader)):
            
            visual_features_1 = self.vpt_model(aug_1)
            visual_features_1 = visual_features_1 / visual_features_1.norm(dim=-1, keepdim=True)
            visual_features_2 = self.vpt_model(aug_2)
            visual_features_2 = visual_features_2 / visual_features_2.norm(dim=-1, keepdim=True)

            # Define positive and negative pairs
            concat_aug_images, similarity_labels = prepare_data_ssl_loss(visual_features_1, 
                                                                         visual_features_2)
            loss = self.ssl_loss(concat_aug_images, similarity_labels) 

            total_loss += loss.item()
            
            accelerator.wait_for_everyone()
            # if accelerator.is_local_main_process:
            #     log.info(f"Loss (batch): {loss}")
            loss = loss / accum_iter 

            accelerator.backward(loss)
            
            if ((i + 1) % accum_iter == 0) or (i + 1 == len(unsupervised_train_loader)):
                self.vpt_optimizer.step()
                self.vpt_model.zero_grad()
        
        accelerator.wait_for_everyone()

        current_lr = self.vpt_scheduler.get_last_lr()
        self.vpt_scheduler.step()

        unwrapped_model = accelerator.unwrap_model(self.vpt_model)
        epoch_parameters = [unwrapped_model.prefix.detach().cpu().numpy(), 
                            unwrapped_model.image_pos_emb.detach().cpu().numpy()]
        
        return loss, total_loss, epoch_parameters

    def train_adapter(self):
        pass

    def _train_supervised_epoch(self):
        pass

    def train(self, train_labeled_files, 
              unlabeled_data, val_labeled_files,
              data_folder):
        
        """ Step 1: train the visual prompts using 
        unlabeled data from seen and unseen classes
        """
        self.unsupervised_training_data = list(train_labeled_files) + unlabeled_data
        self.data_folder = data_folder
        
        vpt_prompts = self.train_visual_prompts()

        accelerator.wait_for_everyone()

        """ Step 2: freeze the vpt_model with the last prompt from
        previous unsupervised training
        """
        self.vpt_model.prefix = torch.nn.Parameter(torch.tensor(vpt_prompts[0]))
        self.vpt_model.image_pos_emb = torch.nn.Parameter(torch.tensor(vpt_prompts[1]))

        for param in self.vpt_model.parameters():
            param.requires_grad = False
        

        return vpt_prompts
        
        




    