from tqdm import tqdm
import logging
import numpy as np
import pandas as pd

import clip
import torch
from torch import nn
from PIL import Image
from .coop_baseline import TextPrefixModel
from accelerate import Accelerator
accelerator = Accelerator()

from ..utils import seed_worker, composed_transform
from ..models import CustomTextEncoder, make_scheduler

g = torch.Generator()
g.manual_seed(0)

log = logging.getLogger(__name__)

class TPTModel(nn.Module):
    def __init__(self, initial_prefix, 
                 text_encoder, classes, 
                 temperature=0.07, device='cpu'):
        super(TextPrefixModel, self).__init__()
        self.device = device
        self.initialized_prefix = initial_prefix
        self.classes = classes
        
        self.prefix = nn.Parameter(initial_prefix, requires_grad=True)
        self.text_encoder = text_encoder

    def forward(self):
        out = self.text_encoder(self.prefix, self.classes)
        out = out / out.norm(dim=-1, keepdim=True)

        return out

class UnsupervisedLoss(torch.nn.Module):
    def __init__(self):
        super(UnsupervisedLoss, self).__init__()

    # custom loss function defined in paper
    def forward(self, preds):
        avg_preds = preds.mean(0)
        avg_preds = avg_preds / torch.sum(avg_preds) # renormalize due to rounding errors
        try:      
            distribution = torch.distributions.Categorical(probs=avg_preds, validate_args=False)
        except:
            raise Exception("{}".format(sum(avg_preds)))
        entropy = distribution.entropy()
        return entropy

class TptBaseline(object):
    def __init__(self, config, 
                 classes, seen_classes, unseen_classes,
                 device):

        self.config = config
        
        self.classes = classes
        self.seen_classes = seen_classes
        self.unseen_classes = unseen_classes

        self.device = device
        self.clip_model, self.transform = clip.load(self.config.VIS_ENCODER, 
                                                    device=self.device)
        self.template = self.config.PROMPT_TEMPLATE

        self.transform_base = ['RandomResizedCrop', 'RandomHorizontalFlip', 'RandomAffine1']
        self.transform_strong = ['RandomGaussianBlur', 'RandomColorJitter', 'RandomResizedCrop', 'RandomGrayscale', 'RandomColorJitter']

        if torch.cuda.is_available():
            self.text_encoder = CustomTextEncoder(self.clip_model, self.device, 
                                                  torch.float16).to(self.device)
        else:
            self.text_encoder = CustomTextEncoder(self.clip_model, self.device,
                                                  torch.half).to(self.device)

        print("Freeze text encoder.")
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        # Get encoding of prefix template
        template_embedding = self.clip_model.token_embedding(clip.tokenize(self.template).to(device)).to(device)
        eoss = clip.tokenize(self.template).to(device)
        template_embedding = template_embedding[:, 1:int(eoss.argmax())] #exclude eoss and soss

        self.model = TextPrefixModel(template_embedding, self.text_encoder,
                                 [' '.join(c.split('_')) for c in self.classes],
                                 device=self.device).to(self.device)

        for i, parameter in enumerate(self.model.parameters()):
            if parameter.requires_grad:
                log.info(f"Shape of parameters {i}: {parameter.shape}")

        if self.config.OPTIM == 'SGD':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), 
                                             lr=.005)

        self.loss_func = UnsupervisedLoss()

    def select_confident_samples(self, logits, top):
        batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
        idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * top)]
        return logits[idx]

    def create_augmentations(self, example, strength='strong'):
        augmentation = composed_transform(strength, transform_base=self.transform_base, transform_strong=self.transform_strong)
        return [example] + [augmentation(example) for _ in range(self.config.NUM_AUGMENTATIONS - 1)]

    def single_example_inference(self, test_example, standard_zsl=False):
        self.model.prefix = nn.Parameter(self.model.initialized_prefix.detach().clone())
        if self.config.OPTIM == 'SGD':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), 
                                             lr=0.005)
        # self.optimizer.zero_grad()
        text_features = self.model()
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        augmentations = self.create_augmentations(test_example)

        # create loader of with one batch containing all augmentations
        augmentation_loader = torch.utils.data.DataLoader(augmentations,
                                                   batch_size=self.config.NUM_AUGMENTATIONS,
                                                   shuffle=False, worker_init_fn=seed_worker,
                                                   generator=g)

        img = next(iter(augmentation_loader)).squeeze()

        with torch.no_grad():
            image_features = self.clip_model.encode_image(img)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits        
        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        logits = self.select_confident_samples(logits, .1)
        distribution = logits.softmax(dim=-1)

        loss = self.loss_func(distribution)
        accelerator.wait_for_everyone()
        accelerator.backward(loss)
        self.optimizer.step()

        # with updated weights, do inference again on the original image
        text_features = self.model()
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        with torch.no_grad():
            image_features = self.clip_model.encode_image(test_example)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logit_scale = self.clip_model.logit_scale.exp()
            logits = logit_scale * image_features @ text_features.t()

        return torch.argmax(logits, 1)


    def test_predictions(self, data, standard_zsl=False):
        
        # Declare the data pre processing
        data.transform = self.transform
        # Define the data loader
        test_loader = torch.utils.data.DataLoader(data, batch_size=1) #TODO: this ok?
        
        accelerator.wait_for_everyone()
        self.model, test_loader = accelerator.prepare(self.model, test_loader)

        if standard_zsl:
            self.model.classes =  [' '.join(c.split('_')) for c in self.unseen_classes]
        else:
            self.model.classes =  [' '.join(c.split('_')) for c in self.classes]

        log.info(f"Start inference for test data")
        predictions = []
        images = []
        for img, _, _, img_path in tqdm(test_loader):
            pred = self.single_example_inference(img)
            if standard_zsl:
                predictions += [self.unseen_classes[pred]]
            else:
                predictions += [self.classes[pred]]
            images += [i for i in img_path]

        accelerator.wait_for_everyone()

        predictions_outputs = accelerator.gather(predictions)
        image_outputs = accelerator.gather(images)

        df_predictions = pd.DataFrame({'id': image_outputs, 
                                       'class': predictions_outputs})
                                       
        return df_predictions