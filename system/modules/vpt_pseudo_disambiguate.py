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
from ..modules import VPTPseudoBaseline

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

class VPTPseudoDisambiguate(VPTPseudoBaseline):
    def __init__(self, config, label_to_idx, 
                 classes, seen_classes, unseen_classes,
                 device):
        super().__init__(config, label_to_idx, classes, 
                         seen_classes, unseen_classes,
                         device)
        
        self.loss_disambiguate = self.disambiguation_loss
        log.info(f"Verify subclass work: {self.loss_disambiguate}")

    def disambiguation_loss(self, logits, labels):
        """ This loss computes the probability mass on the
        opposite set of classes for each sample.
        
        :param logits: continuous vector
        :param labels: class ids
        """
        softmax = nn.Softmax(dim=1)
        logits = softmax(logits)
        #log.info(f"LOGITS: {logits}")
        seen_ids = [self.label_to_idx[c] for c in self.seen_classes]
        unseen_ids = [self.label_to_idx[c] for c in self.unseen_classes]

        # Get indices of unseen and seen samples in the batch
        unseen_samples = [] 
        seen_samples = []
        for idx, l in enumerate(labels):
            if l in unseen_ids:
                unseen_samples.append(idx)
            elif l in seen_ids:
                seen_samples.append(idx)
        # Get logit sums on unseen samples
        if unseen_samples:
            error_unseen = torch.sum(logits[unseen_samples][:, seen_ids])
        else:
            error_unseen = 0
        if seen_samples:
            error_seen = torch.sum(logits[seen_samples][:,unseen_ids])
        else:
            error_seen = 0
        return error_unseen + error_seen

        
    def _train_epoch(self, loss, total_loss, train_loader, accum_iter, epoch, only_unlabelled=False):

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
            loss_ce = self.loss_func(logits, labs)
            loss_dis = self.loss_disambiguate(logits, labs)
            loss = loss_ce + loss_dis
            #log.info(f"LOSS: {loss}")
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


