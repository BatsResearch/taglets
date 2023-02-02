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

from ..data import CustomDataset
from ..utils import seed_worker, pseudolabel_top_k
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

class VPTPseudoBaseline(object):
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
        self.model = ImagePrefixModel(vis_initial_prefix,
                                      initial_pos_emb,
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
        
    
    def train(self, train_data, unlabeled_data, val_data, data_folder):

        train_unseen_dataset = pseudolabel_top_k(self.config.N_PSEUDOSHOTS,
                                                 self.config.PROMPT_TEMPLATE,
                                                 unlabeled_data,
                                                 self.unseen_classes,
                                                 self.transform,
                                                 self.clip_model,
                                                 self.label_to_idx,
                                                 self.device)
        
        unseen_imgs = train_unseen_dataset.filepaths#[f.split('/')[-1] for f in train_unseen_dataset.filepaths]
        unseen_labs = train_unseen_dataset.labels

        # log.info(f"{self.label_to_idx}")
        # log.info(f"{set(unseen_labs)}")
        # log.info(f"unseen classes: {self.unseen_classes}")
        seen_imgs = train_data.filepaths#[f.split('/')[-1] for f in train_data.filepaths]
        seen_labs = [self.label_to_idx[l] for l in train_data.labels]
        log.info(f"unseen classes: {self.seen_classes}")
        log.info(f"{set(seen_labs)}")
        # Declare the data pre processing for train and validation data

        log.info(f"Training unseen data labels: {len(unseen_labs)}")
        log.info(f"Training unseen data images: {len(unseen_imgs)}")
        log.info(f"Training seen data labels: {len(seen_labs)}")
        log.info(f"Training seen data images: {len(seen_imgs)}")
        train_data.filepaths = list(unseen_imgs) + list(seen_imgs)
        train_data.labels = list(unseen_labs) + list(seen_labs)
        train_data.label_id = True
        # train_data = CustomDataset(list(unseen_imgs) + list(seen_imgs), data_folder, 
        #                            transform=None, augmentations=None, 
        #                            train=True, labels=list(unseen_labs) + list(seen_labs),
        #                            label_id=True,
        #                            label_map=self.label_to_idx)

        train_data.transform = self.transform
        val_data.transform = self.transform
        log.info(f"Training data size: {len(list(unseen_imgs) + list(seen_imgs)) == len(train_data.filepaths)}")

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

        # Define text queries
        prompts = [f"{self.template}{' '.join(i.split('_'))}" \
                        for i in self.classes]
        log.info(f"Number of prompts: {len(prompts)}")
        # Encode text
        with torch.no_grad():
            text = clip.tokenize(prompts).to(self.device)
            text_features = self.clip_model.encode_text(text)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        predictions = []
        images = []
        labels = []
        for i, (img, _, _, label, img_path) in enumerate(tqdm(train_loader)):
            image_features = self.model(img)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # cosine similarity as logits        
            logit_scale = self.clip_model.logit_scale.exp()
            logits = logit_scale * image_features @ text_features.t()

            idx_preds = torch.argmax(logits, dim=1)
            real_preds = [self.classes[i.item()] for i in idx_preds]
            
            predictions += real_preds
            labels += [self.classes[i.item()] for i in label]
            images += [i for i in img_path]

            labs = torch.tensor([l for l in label])
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
        epoch_parameters = [unwrapped_model.prefix.detach().cpu().numpy(), 
                            unwrapped_model.image_pos_emb.detach().cpu().numpy()]

        return loss, total_loss, epoch_parameters

    def _run_validation(self, val_loader, epoch):

        # Define text queries
        prompts = [f"{self.template}{' '.join(i.split('_'))}" \
                        for i in self.seen_classes]
        log.info(f"Number of prompts: {len(prompts)}")
        # Encode text
        with torch.no_grad():
            text = clip.tokenize(prompts).to(self.device)
            text_features = self.clip_model.encode_text(text)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        predictions = []
        labels = []
        for img, _, _, label, img_path in tqdm(val_loader):
            image_features = self.model(img)
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

        # Define text queries
        if standard_zsl:
            prompts = [f"{self.template}{' '.join(i.split('_'))}" \
                        for i in self.unseen_classes]
        else:
            prompts = [f"{self.template}{' '.join(i.split('_'))}" \
                        for i in self.classes]
        log.info(f"Number of prompts: {len(prompts)}")

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

        accelerator.wait_for_everyone()

        predictions_outputs = accelerator.gather(predictions)
        image_outputs = accelerator.gather(images)


        df_predictions = pd.DataFrame({'id': image_outputs, 
                                       'class': predictions_outputs})

        return df_predictions
