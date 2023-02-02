from tqdm import tqdm
import logging
import numpy as np
import pandas as pd

import clip
import torch
from PIL import Image
from accelerate import Accelerator
accelerator = Accelerator()


g = torch.Generator()
g.manual_seed(0)

log = logging.getLogger(__name__)

class ClipBaseline(object):
    def __init__(self, config, 
                 classes, seen_classes, unseen_classes,
                 device):
        self.config = config
        self.classes = classes
        self.seen_classes = seen_classes
        self.unseen_classes = unseen_classes

        self.device = device
        self.model, self.transform = clip.load(self.config.VIS_ENCODER, 
                                               device=self.device)
        self.template = self.config.PROMPT_TEMPLATE

    def test_predictions(self, data, 
                         standard_zsl=False):
        """
        :param data: test dataset
        :param standard_zsl: True if standard preds
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
        # Encode text
        text = clip.tokenize(prompts).to(self.device)
        text_features = self.model.encode_text(text)

        log.info(f"Start inference for test data")
        predictions = []
        images = []
        # Iterate through data loader
        for img, _, _, img_path in tqdm(test_loader):
            with torch.no_grad():
                logits_per_image, logits_per_text = self.model(img, text)
                probs = logits_per_image.softmax(dim=-1)
                idx_preds = torch.argmax(probs, dim=1)
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
        