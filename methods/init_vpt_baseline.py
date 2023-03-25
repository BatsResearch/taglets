import copy
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

from methods import VPTBaseline
from models import CustomImageEncoder, ImagePrefixModel
from utils import make_scheduler, seed_worker, pseudolabel_top_k


g = torch.Generator()
g.manual_seed(0)

log = logging.getLogger(__name__)

class InitVPTBaseline(VPTBaseline):
    def __init__(
        self, 
        config, 
        label_to_idx,
        classes, 
        seen_classes, 
        unseen_classes, 
        init_param,
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
        :param kind: 'init', 'mix', 'cat'
        :param device: device in use
        """
        super().__init__(
            config, label_to_idx, classes, seen_classes, unseen_classes, device
        )
        
        original_init = copy.deepcopy(init_param)
        self.balance_weight = self.config.N_PSEUDOSHOTS*(len(self.unseen_classes))
        if kind == 'mix' or kind == 'cat':
            visual_transformer = self.clip_model.visual
            self.image_encoder = CustomImageEncoder(
                visual_transformer,
                init_prefix=original_init, 
                alpha=self.config.ALPHA, 
                kind=kind,
            ).to(self.device)
            log.info(f"Freeze visual encoder.")
            for param in self.image_encoder.parameters():
                param.requires_grad = False
        
        self.vis_initial_prefix = init_param # self.initialize_model_parameters(visual_transformer)
        self.config.EPOCHS = self.config.adapt_EPOCHS

    def initialize_model_parameters(self, visual_transformer=None):
        """This function defines the models' parameters initialization."""

        # Initialize prefix and pos embeddings for visual prompts
        width = visual_transformer.class_embedding.size()[0]
        scale = width**-0.5
        if self.config.VIS_PREFIX_INIT == "normal":
            vis_initial_prefix = scale * torch.randn(self.config.PREFIX_SIZE, width)

        elif self.config.VIS_PREFIX_INIT == "uniform":
            val = math.sqrt(6.0 / float(3 * reduce(mul, (16, 16), 1) + width))  # noqa
            vis_initial_prefix = torch.zeros(self.config.PREFIX_SIZE, width)
            vis_initial_prefix = scale * nn.init.uniform_(vis_initial_prefix, -val, val)

        return vis_initial_prefix

    def create_training_dataset(self, train_data, unlabeled_data=None):
        """This function create the dataset for training. Specifically, it
        merges pseudo-labels for unseen data and labeled data for seen classes.

        :param train_data: Dataset object - training seen classes (defined in zsl_jpl line 323)
        :param unlabeled_data: Dataset object - dataset of unlabeled data for
                               unseen classes (defined in zsl_jpl line 328)
        """
        # Get pseudo-labels for unlabeled data from unseen classes
        train_unseen_dataset = pseudolabel_top_k(
            self.config.DATASET_NAME,
            self.config.N_PSEUDOSHOTS,
            self.config.PROMPT_TEMPLATE,
            unlabeled_data,
            self.unseen_classes,
            self.transform,
            self.clip_model,
            self.label_to_idx,
            self.device,
            self.config.VIS_ENCODER,
            self.config.SPLIT_SEED,
        )
        
        # Define the lists of traiing data from seen and unseen classes
        unseen_imgs = train_unseen_dataset.filepaths
        unseen_labs = train_unseen_dataset.labels

        # Use a portion of the pseudo-labeled data to build a validation set
        if self.config.N_PSEUDOSHOTS >= 10:
            np.random.seed(self.config.validation_seed)
            train_indices = np.random.choice(
                range(len(unseen_imgs)),
                size=int(len(unseen_imgs) * self.config.ratio_train_val),
                replace=False,
            )
            val_indices = list(
                set(range(len(unseen_imgs))).difference(set(train_indices))
            )

            self.val_unseen_files = np.array(unseen_imgs)[val_indices]
            self.val_unseen_labs = np.array(unseen_labs)[val_indices]

            unseen_imgs = list(np.array(unseen_imgs)[train_indices])
            unseen_labs = list(np.array(unseen_labs)[train_indices])

        else:
            self.val_unseen_files = None
            self.val_unseen_labs = None

        # self.balance_param = len(seen_imgs) / len(unseen_imgs)

        train_data.filepaths = list(unseen_imgs)
        train_data.labels = list(unseen_labs)
        train_data.label_id = True

        return train_data

    def train(
        self,
        train_data,
        val_data,
        unlabeled_data=None,
        only_unlabelled=False,
        only_seen=False,
    ):
        """This function defines the training of self.model.

        :param train_data: Dataset object - training dataset of labeled data for
                           seen classes (defined in zsl_jpl line 323)
        :param val_data: Dataset object - validation dataset of labeled data for
                         seen classes (defined in zsl_jpl line 334)
        :param unlabeled_data: Dataset object - dataset of unlabeled data for
                               unseen classes (defined in zsl_jpl line 328)
        :param only_unlabelled: boolean. It is True if the training only involves
                                pseudo-labeled unseen data
        """

        # Define training dataset
        self.create_training_dataset(train_data, unlabeled_data)

        self.define_model()
        # Declare the data pre processing for train and validation data
        train_data.transform = self.transform
        val_data.transform = self.transform

        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            worker_init_fn=seed_worker,
            generator=g,
        )
        if self.val_unseen_files is not None:
            unseen_imgs = list(self.val_unseen_files)
            unseen_labs = list(self.val_unseen_labs)

            val_data.filepaths = list(unseen_imgs)
            val_data.labels = list(unseen_labs)
            val_data.label_id = True

        val_loader = torch.utils.data.DataLoader(
            val_data, batch_size=self.config.BATCH_SIZE
        )

        accelerator.wait_for_everyone()

        self.model, self.optimizer, train_loader, val_loader = accelerator.prepare(
            self.model, self.optimizer, train_loader, val_loader
        )

        best_val_accuracy = 0
        best_prompt = None
        loss = None
        if val_loader is not None:
            log.info(f"Size of validation dataset: {len(val_data.filepaths)}")

        for epoch in range(self.config.EPOCHS):
            log.info(f"Run Epoch {epoch}")
            total_loss = 0
            accum_iter = self.config.ACCUMULATION_ITER

            loss, total_loss, epoch_parameters = self._train_epoch(
                loss,
                total_loss,
                train_loader,
                accum_iter,
                epoch,
                only_unlabelled=True,
                only_seen=only_seen,
            )
            accelerator.wait_for_everyone()
            if accelerator.is_local_main_process:
                log.info(f"Loss Epoch {epoch}: {total_loss/(len(train_loader))}")

            accelerator.free_memory()

            if val_loader is not None:
                val_accuracy = self._run_validation(val_loader, only_unlabelled=True)
                log.info(f"Validation accuracy after Epoch {epoch}: {val_accuracy}")
                if val_accuracy >= best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    best_prompt = epoch_parameters
            else:
                best_val_accuracy = None
                best_prompt = epoch_parameters

        return best_val_accuracy, best_prompt

    def define_textual_prompts(self, only_unlabelled=False, validation=False):
        """This function returns the textual prompts. You can modify the list
        of classes of interest.

        :param only_unlabelled: boolean. It is True if the training only involves
                                pseudo-labeled unseen data
        """
        # return [f"{self.template}{' '.join(i.split('_'))}" \
        #                     for i in self.seen_classes]
        if only_unlabelled:
            return [self.template.format(" ".join(i.split("_"))) for i in self.unseen_classes]
        else:
            return [self.template.format(" ".join(i.split("_"))) for i in self.classes]

    def reindex_predicted_labels(self, idx_preds, only_unlabelled=False):
        """This function returns the correct index of predictions to compute
        model's accuracy.

        :param idx_pred: list of predictions ids
        :param only_unlabelled: boolean. It is True if the training only involves
                                pseudo-labeled unseen data
        """
        if only_unlabelled:
            return [self.unseen_classes[i.item()] for i in idx_preds]
        else:
            return [self.classes[i.item()] for i in idx_preds]

    def reindex_true_labels(self, label, only_unlabelled=False):
        """This function returns the correct index of true labels.

        :param label: list of labels from data loader
        :param only_unlabelled: boolean. It is True if the training only involves
                                pseudo-labeled unseen data
        """

        if only_unlabelled:
            return torch.tensor(
                [self.unseen_classes.index(self.classes[l.item()]) for l in label]
            )
        else:
            return torch.tensor(
                [self.classes.index(self.classes[l.item()]) for l in label]
            )

    def define_loss_function(self, logits, labs, teacher=False):
        return self.loss_func(logits, labs)

    def training_model(self, img, teacher=False):
        """This function allows to customize the model to use while trainig

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

    def _train_epoch(
        self,
        loss,
        total_loss,
        train_loader,
        accum_iter,
        epoch,
        only_unlabelled=False,
        teacher=False,
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
        :param teachet: boolean. Added to use this function in more subclasses
        """

        # Define text queries
        prompts = self.define_textual_prompts(only_unlabelled=False)
        log.info(f"Number of prompts: {len(prompts)}")

        # Encode text
        with torch.no_grad():
            text = clip.tokenize(prompts).to(self.device)
            text_features = self.clip_model.encode_text(text)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        predictions = []
        labels = []
        for i, (img, _, _, label, img_path) in enumerate(train_loader):
            image_features = self.training_model(img, teacher)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # cosine similarity as logits
            logit_scale = self.clip_model.logit_scale.exp()
            logits = logit_scale * image_features @ text_features.t()
            idx_preds = torch.argmax(logits, dim=1)
            #log.info(f"variables idx_preds: {idx_preds}")
            #log.info(f"variables only_unlabelled: {only_unlabelled}")
            real_preds = self.reindex_predicted_labels(idx_preds, only_unlabelled=False)

            predictions += real_preds
            labels += [self.classes[i.item()] for i in label]

            labs = self.reindex_true_labels(label, only_unlabelled=False)
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

        predictions = torch.tensor([self.label_to_idx[p] for p in predictions]).to(
            self.device
        )
        labels = torch.tensor([self.label_to_idx[l] for l in labels]).to(self.device)

        predictions_outputs = accelerator.gather(predictions)
        labels_outputs = accelerator.gather(labels)

        # Get harmonic mean
        idx_seen = [self.label_to_idx[c] for c in self.seen_classes]
        seen_true = [i for i, c in enumerate(labels_outputs) if c in idx_seen]
        seen_preds = predictions_outputs[seen_true]
        seen_labs = labels_outputs[seen_true]
        seen_accuracy = torch.sum(seen_preds == seen_labs) / len(seen_true)

        idx_unseen = [self.label_to_idx[c] for c in self.unseen_classes]
        unseen_true = [i for i, c in enumerate(labels_outputs) if c in idx_unseen]
        unseen_preds = predictions_outputs[unseen_true]
        unseen_labs = labels_outputs[unseen_true]
        unseen_accuracy = torch.sum(unseen_preds == unseen_labs) / len(unseen_true)

        if only_unlabelled:
            accuracy = unseen_accuracy
            log.info(f"Training UNSEEN accuracy after Epoch {epoch}: {unseen_accuracy}")
        else:
            if only_seen:
                accuracy = seen_accuracy
                log.info(f"Training SEEN accuracy after Epoch {epoch}: {accuracy}")
            else:
                accuracy = st.hmean([unseen_accuracy.cpu(), seen_accuracy.cpu()])

                # accuracy = torch.sum(predictions_outputs == labels_outputs)/len(predictions_outputs)
                log.info(f"Training SEEN accuracy after Epoch {epoch}: {seen_accuracy}")
                log.info(
                    f"Training UNSEEN accuracy after Epoch {epoch}: {unseen_accuracy}"
                )
                log.info(f"Training HARMONIC accuracy after Epoch {epoch}: {accuracy}")

        self.update_scheduler(teacher)

        unwrapped_model = self.unwrap_model(teacher)
        epoch_parameters = [
            unwrapped_model.prefix.detach().cpu().numpy(),
        ]

        return loss, total_loss, epoch_parameters

    def _run_validation(
        self, 
        val_loader, 
        only_unlabelled=False, 
        teacher=False, 
        only_seen=False,
    ):
        """This function computes the validation accuracy on labeled seen data.

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
        for img, _, _, label, img_path in val_loader:
            image_features = self.training_model(img, teacher)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            logit_scale = self.clip_model.logit_scale.exp()
            logits = logit_scale * image_features @ text_features.t()

            idx_preds = torch.argmax(logits, dim=1)
            if self.val_unseen_files is not None:
                real_preds = [self.unseen_classes[i.item()] for i in idx_preds]
            else:
                real_preds = [self.unseen_classes[i.item()] for i in idx_preds]

            predictions += real_preds
            labels += [self.classes[i.item()] for i in label]

        accelerator.wait_for_everyone()

        predictions = torch.tensor([self.label_to_idx[p] for p in predictions]).to(
            self.device
        )
        labels = torch.tensor([self.label_to_idx[l] for l in labels]).to(self.device)

        predictions_outputs = accelerator.gather(predictions)
        labels_outputs = accelerator.gather(labels)

        if len(prompts) < len(self.classes):
            accuracy = torch.sum(predictions_outputs == labels_outputs) / len(
                predictions_outputs
            )
        else:
            # Get harmonic mean
            idx_seen = [self.label_to_idx[c] for c in self.seen_classes]
            seen_true = [i for i, c in enumerate(labels_outputs) if c in idx_seen]
            seen_preds = predictions_outputs[seen_true]
            seen_labs = labels_outputs[seen_true]
            seen_accuracy = torch.sum(seen_preds == seen_labs) / len(seen_true)

            idx_unseen = [self.label_to_idx[c] for c in self.unseen_classes]
            unseen_true = [i for i, c in enumerate(labels_outputs) if c in idx_unseen]
            unseen_preds = predictions_outputs[unseen_true]
            unseen_labs = labels_outputs[unseen_true]
            unseen_accuracy = torch.sum(unseen_preds == unseen_labs) / len(unseen_true)

            if only_unlabelled:
                accuracy = unseen_accuracy
                log.info(f"Validation SEEN accuracy after Epoch: {unseen_accuracy}")

            else:
                accuracy = st.hmean([unseen_accuracy.cpu(), seen_accuracy.cpu()])
                log.info(f"Validation SEEN accuracy after Epoch: {seen_accuracy}")
                log.info(f"Validation UNSEEN accuracy after Epoch: {unseen_accuracy}")
                log.info(f"Validation HARMONIC accuracy after Epoch: {accuracy}")

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

        accelerator.wait_for_everyone()

        self.model, test_loader = accelerator.prepare(self.model, test_loader)

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
                image_features = self.model(img)
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
