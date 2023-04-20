import logging

import clip
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm


g = torch.Generator()
g.manual_seed(0)

log = logging.getLogger(__name__)


class ClipBaseline(object):
    def __init__(
        self, config, label_to_idx, classes, seen_classes, unseen_classes, device
    ):
        """This class is CLIP model.

        :param config: class object with model configurations in the
                       file models_config/clip_baseline_config.yml
        :param classes: list of class names
        :param seen_classes: list on seen classes' names
        :param unseen_classes: list on unseen classes' names
        :param device: device in use

        """
        self.config = config
        self.classes = classes
        self.seen_classes = seen_classes
        self.unseen_classes = unseen_classes
        self.label_to_idx = label_to_idx

        self.device = device
        self.model, self.transform = clip.load(
            self.config.VIS_ENCODER, device=self.device
        )
        self.template = self.config.PROMPT_TEMPLATE

    def test_predictions(self, data, standard_zsl=False):
        """
        :param data: test dataset
        :param standard_zsl: True if standard preds
        """

        # Declare the data pre processing
        data.transform = self.transform
        # Define the data loader
        test_loader = torch.utils.data.DataLoader(
            data, batch_size=self.config.BATCH_SIZE
        )

        # Define text queries
        if standard_zsl:
            # prompts = [f"{self.template}{' '.join(i.split('_'))}" \
            #             for i in self.unseen_classes]
            prompts = [
                self.template.format(" ".join(i.split("_")))
                for i in self.unseen_classes
            ]
        else:
            # prompts = [f"{self.template}{' '.join(i.split('_'))}" \
            #             for i in self.classes]
            prompts = [
                self.template.format(" ".join(i.split("_"))) for i in self.classes
            ]

        log.info(f"Number of prompts: {len(prompts)}")
        # Encode text
        text = clip.tokenize(prompts).to(self.device)
        text_features = self.model.encode_text(text)

        log.info(f"Start inference for test data")
        test_files = [f.split("/")[-1] for f in test_loader.dataset.filepaths]

        predictions = []
        images = []
        prob_preds = []
        # Iterate through data loader
        for img, _, _, img_path in tqdm(test_loader):
            with torch.no_grad():
                logits_per_image, logits_per_text = self.model(
                    img.to(self.device), text.to(self.device)
                )
                probs = logits_per_image.softmax(dim=-1)
                idx_preds = torch.argmax(probs, dim=1)

                predictions += [self.classes[i] for i in idx_preds]
                images += [i for i in img_path]
                prob_preds += [logits_per_image]

        prob_preds = torch.cat(prob_preds, axis=0).detach().to('cpu')

        log.info(f"Number of images: {len(images)}")
        log.info(f"Number of images: {len(predictions)}")
        log.info(f"Number of probs: {prob_preds.size()}")

        df_predictions = pd.DataFrame({"id": images, "class": predictions})

        return df_predictions, images, predictions, prob_preds
