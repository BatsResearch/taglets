import logging

import clip
import numpy as np
import pandas as pd
import scipy.stats as st
import torch
from accelerate import Accelerator
from PIL import Image
from torch import nn

accelerator = Accelerator()

from methods import InitVPTBaseline
from models import CustomImageEncoder, ImagePrefixModel
from utils import make_scheduler, seed_worker, pseudolabel_top_k


g = torch.Generator()
g.manual_seed(0)

log = logging.getLogger(__name__)

class SeparateVPTBaseline(InitVPTBaseline):
    def __init__(
        self, 
        config, 
        label_to_idx,
        classes, 
        seen_classes, 
        unseen_classes, 
        seen_param,
        unseen_param,
        device,
        kind='init',
    ):
        """This class defines Coop's training and evaluation.

        :param config: dictionaries of prameters in models_config/vpt_baseline_config.yml
        :param label_to_idx: dictionary (key, value):(class name, id)
        :param classes: list of class names
        :param seen_classes: list of seen classes' names
        :param unseen_classes: list of unseen classes' names
        :param init_param: initial parameters for the prompts
        :param kind: 'init', 'mix'
        :param device: device in use
        """
        super().__init__(
            config, 
            label_to_idx, 
            classes, 
            seen_classes, 
            unseen_classes, 
            init_param=seen_param,
            device=device,
            kind='init',
        )
        
        self.vis_seen_prefix = seen_param
        self.vis_unseen_prefix = unseen_param
        self.config.EPOCHS = self.config.adapt_EPOCHS

        self.define_models()

    def define_models(self):
        # Define model seen
        self.model_seen = ImagePrefixModel(
            self.vis_seen_prefix,
            self.image_encoder,
            device=self.device,
        ).to(self.device)

        for i, parameter in enumerate(self.model_seen.parameters()):
            if parameter.requires_grad:
                log.info(f"Shape of parameters {i}: {parameter.shape}")

        # Define model unseen
        self.model_unseen = ImagePrefixModel(
            self.vis_unseen_prefix,
            self.image_encoder,
            device=self.device,
        ).to(self.device)

        for i, parameter in enumerate(self.model_unseen.parameters()):
            if parameter.requires_grad:
                log.info(f"Shape of parameters {i}: {parameter.shape}")

    def test_predictions(self, data, standard_zsl=False):
        """This function computes predictions on test data.

        :param data: Dataset object - test dataset
        """

        # Declare the data pre processing
        data.transform = self.transform
        # Define the data loader
        test_loader = torch.utils.data.DataLoader(
            data, batch_size=self.config.BATCH_SIZE
        )

        accelerator.wait_for_everyone()

        self.model_seen, self.model_unseen, test_loader = accelerator.prepare(self.model_seen, 
        self.model_unseen,
        test_loader,
        )

        # Define text queries
        if standard_zsl:
            prompts = [
                self.template.format(" ".join(i.split("_")))
                for i in self.unseen_classes
            ]
        else:
            prompts = [
                self.template.format(" ".join(i.split("_"))) for i in self.classes
            ]

        log.info(f"Number of prompts: {len(prompts)}")
        # This is required for distributed training
        test_files = [f.split("/")[-1] for f in test_loader.dataset.filepaths]

        # Encode text
        text = clip.tokenize(prompts).to(self.device)
        text_features = self.clip_model.encode_text(text)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        log.info(f"Start inference for test data")
        predictions = []
        images = []
        for img, _, _, img_path in test_loader:
            with torch.no_grad():
                image_features_seen = self.model_seen(img)
                image_features_seen = image_features_seen / image_features_seen.norm(
                    dim=-1, keepdim=True
                )

                image_features_unseen = self.model_unseen(img)
                image_features_unseen = image_features_unseen / image_features_unseen.norm(
                    dim=-1, keepdim=True
                )
                # cosine similarity as logits

            #log.info(f"shape seen: {image_features_seen.shape}")
            #log.info(f"shape unseen: {image_features_unseen.shape}")
            #image_features = self.config.ALPHA*image_features_unseen + (1 - self.config.ALPHA)*image_features_unseen
            image_features = image_features_unseen + (self.config.N_PSEUDOSHOTS*len(self.unseen_classes))*image_features_seen
            image_features = image_features / image_features.norm(
                    dim=-1, keepdim=True
            )
            logit_scale = self.clip_model.logit_scale.exp()
            logits = logit_scale * image_features @ text_features.t()
            idx_preds = torch.argmax(logits, dim=1)

            if standard_zsl:
                predictions += [self.unseen_classes[i] for i in idx_preds]
            else:
                predictions += [self.classes[i] for i in idx_preds]

            images += [i for i in img_path]

        predictions = torch.tensor([self.label_to_idx[p] for p in predictions]).to(
            self.device
        )
        images = torch.tensor([test_files.index(img) for img in images]).to(self.device)

        accelerator.wait_for_everyone()

        predictions_outputs = accelerator.gather(predictions)
        image_outputs = accelerator.gather(images)

        predictions_outputs = [self.classes[p] for p in predictions_outputs]
        image_outputs = [test_files[i] for i in image_outputs]

        df_predictions = pd.DataFrame(
            {"id": image_outputs, "class": predictions_outputs}
        )
        df_predictions.drop_duplicates(subset=["id", "class"], inplace=True)

        return df_predictions
