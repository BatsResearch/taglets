import logging

import clip
import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from PIL import Image
from torch import nn

accelerator = Accelerator()

from models import CustomTextEncoder, TextPrefixModel
from methods import TrainingStrategy
from utils import make_scheduler, seed_worker

g = torch.Generator()
g.manual_seed(0)

log = logging.getLogger(__name__)


class TextualPrompt(TrainingStrategy):
    def __init__(
        self,
        config,
        label_to_idx,
        classes,
        seen_classes,
        unseen_classes,
        device,
    ):
        """This class define Coop's training and evaluation.

        :param config: dictionaries of prameters in models_config/coop_baseline_config.yml
        :param label_to_idx: dictionary (key, value):(class name, id)
        :param classes: list of class names
        :param seen_classes: list of seen classes' names
        :param unseen_classes: list of unseen classes' names
        :param device: device in use
        """

        super().__init__(
            config, label_to_idx, classes, seen_classes, unseen_classes, device
        )

        # Build dictionaries to correctly label model's predictions
        seen_to_idx = {c: idx for idx, c in enumerate(self.seen_classes)}
        self.idx_to_real = {
            seen_to_idx[c]: self.label_to_idx[c] for c in self.seen_classes
        }
        self.real_to_idx = {
            self.label_to_idx[c]: seen_to_idx[c] for c in self.seen_classes
        }

        # Load custom encoder
        self.declare_custom_encoder()
        log.info(f"Custom Encoder: {self.image_encoder}.")
        # Initialize prompt parameters
        self.initialize_prompts_parameters()

    def _train_epoch(
        self, 
        loss, 
        total_loss, 
        train_loader, 
        accum_iter, 
        epoch, 
        only_unlabelled=False,
        only_seen=False,
    ):
        """This function defines the training epoch of self.model.

        :param loss: float loss (average across batches)
        :param total_loss: float total loss
        :param train_loader: Dataloader object - training data defined in self.train
        :param accum_iter: number of accumulation steps minimum 1
        :param epoch: current epoch
        :param only_unlabelled: boolean. It is True if the training only involves
                                pseudo-labeled unseen data
        :param only_seen: boolean.  It is True if the training only involves
                                seen data

        """

        predictions = []
        labels = []
        for i, (img, _, _, label, img_path) in enumerate(train_loader):
            text_features = self.model(self.model.module.classes)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            with torch.no_grad():
                image_features = self.clip_model.encode_image(img)
                image_features = image_features / image_features.norm(
                    dim=-1, keepdim=True
                )

            # cosine similarity as logits
            logit_scale = self.clip_model.logit_scale.exp()
            logits = logit_scale * image_features @ text_features.t()

            idx_preds = torch.argmax(logits, dim=1)
            if only_seen:
                real_preds = [self.seen_classes[i.item()] for i in idx_preds]
            else:
                real_preds = [self.classes[i.item()] for i in idx_preds]

            predictions += real_preds
            labels += [self.classes[i.item()] for i in label]

            if only_seen:
                labs = torch.tensor([self.real_to_idx[l.item()] for l in label]).to(
                    self.device
                )
            else:
                labs = torch.tensor([l.item() for l in label]).to(self.device)

            loss = self.define_loss_function(logits, labs)
            total_loss += loss.item()

            accelerator.wait_for_everyone()

            loss = loss / accum_iter
            accelerator.backward(loss)

            # Accumulate grandient
            if ((i + 1) % accum_iter == 0) or (i + 1 == len(train_loader)):
                self.backpropagate()

        accelerator.wait_for_everyone()

        predictions = torch.tensor(
            [self.label_to_idx[p] for p in predictions][: len(train_loader.dataset)]
        ).to(self.device)
        labels = torch.tensor(
            [self.label_to_idx[l] for l in labels][: len(train_loader.dataset)]
        ).to(self.device)

        predictions_outputs = accelerator.gather(predictions)
        labels_outputs = accelerator.gather(labels)

        accuracy = torch.sum(predictions_outputs == labels_outputs) / len(
            predictions_outputs
        )
        log.info(f"Training accuracy after Epoch {epoch}: {accuracy}")

        self.update_scheduler()

        unwrapped_model = self.unwrap_model()
        epoch_parameters = [
            unwrapped_model.prefix.detach().cpu().numpy()
        ]

        return loss, total_loss, epoch_parameters

    def _run_validation(
        self, 
        val_loader,
        only_unlabelled=False, 
        only_seen=False,
    ):
        """This function computes the validation accuracy on labeled seen data.

        :param val_loder: Dataloader object - validation dataset
        """

        predictions = []
        labels = []
        for img, _, _, label, img_path in val_loader:
            self.model.classes = self.seen_classes
            text_features = self.model(self.model.classes)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            with torch.no_grad():
                image_features = self.clip_model.encode_image(img)
                image_features = image_features / image_features.norm(
                    dim=-1, keepdim=True
                )

            logit_scale = self.clip_model.logit_scale.exp()
            logits = logit_scale * image_features @ text_features.t()

            idx_preds = torch.argmax(logits, dim=1)
            real_preds = [self.seen_classes[i.item()] for i in idx_preds]

            predictions += real_preds
            labels += [self.classes[i.item()] for i in label]

        accelerator.wait_for_everyone()

        predictions = torch.tensor(
            [self.label_to_idx[p] for p in predictions][: len(val_loader.dataset)]
        ).to(self.device)
        labels = torch.tensor(
            [self.label_to_idx[l] for l in labels][: len(val_loader.dataset)]
        ).to(self.device)

        predictions_outputs = accelerator.gather(predictions)
        labels_outputs = accelerator.gather(labels)

        accuracy = torch.sum(predictions_outputs == labels_outputs) / len(
            predictions_outputs
        )

        return accuracy

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

        self.model, test_loader = accelerator.prepare(self.model, test_loader)

        if standard_zsl:
            self.model.classes = self.unseen_classes
        else:
            self.model.classes = self.classes

        accelerator.wait_for_everyone()

        # Get prompts
        text_features = self.model(self.model.classes)

        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        log.info(f"TEXT FEATURES SHAPE: {text_features.size()}")

        log.info(f"Start inference for test data")
        # This is required for distributed training
        test_files = [f.split("/")[-1] for f in test_loader.dataset.filepaths]

        predictions = []
        images = []
        for img, _, _, img_path in test_loader:
            with torch.no_grad():
                image_features = self.clip_model.encode_image(img)
                image_features = image_features / image_features.norm(
                    dim=-1, keepdim=True
                )
                # cosine similarity as logits
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
