import copy
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
        # Combine prefix and class embeddings to get the
        # entire prompt representation for the
        # two augmented images
        out = self.image_encoder(x, self.prefix, self.image_pos_emb)
        norm_out = out / out.norm(dim=-1, keepdim=True)

        return out

class TeacherStudent(object):
    def __init__(self, config, label_to_idx, data_folder, 
                 classes, seen_classes, unseen_classes,
                 device):

        self.config = config
        
        self.classes = classes
        self.seen_classes = seen_classes
        self.unseen_classes = unseen_classes
        self.label_to_idx = label_to_idx
        self.data_folder = data_folder

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
        vis_initial_prefix, initial_pos_emb = self.initialize_prompts(visual_transformer)
        # Define models
        self.teacher = ImagePrefixModel(vis_initial_prefix,
                                      initial_pos_emb,
                                      self.image_encoder,
                                      device=self.device).to(self.device)
        for i, parameter in enumerate(self.teacher.parameters()):
            if parameter.requires_grad:
                log.info(f"Shape of teacher parameters {i}: {parameter.shape}")

        if self.config.t_OPTIM == 'SGD':
            self.teacher_optimizer = torch.optim.SGD(self.teacher.parameters(), 
                                             lr=self.config.t_LR, 
                                             weight_decay=self.config.t_DECAY,
                                             momentum=0.9)

        self.teacher_scheduler = make_scheduler(self.teacher_optimizer, self.config)
        self.teacher_loss_func = torch.nn.CrossEntropyLoss()
        
    def initialize_prompts(self, visual_transformer):
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

        return vis_initial_prefix, initial_pos_emb

    def train(self, train_data, unlabeled_data, val_data):
        # 1. Get pseudo-labels
        unlabeled_examples = copy.deepcopy(unlabeled_data)
        num_pseudo_labels = int((len(unlabeled_data) / self.config.STEP_QUANTILE) / len(self.unseen_classes))
        log.info(f"We select {num_pseudo_labels} per each unseen classes.")
        log.info(f"The total number of pseudo-labeled images is {num_pseudo_labels*len(self.unseen_classes)}")
        
        train_unseen_dataset = pseudolabel_top_k(num_pseudo_labels,
                                                 self.config.PROMPT_TEMPLATE,
                                                 unlabeled_data,
                                                 self.unseen_classes,
                                                 self.transform,
                                                 self.clip_model,
                                                 self.label_to_idx,
                                                 self.device)
        # 2. Train teacher with labeled seen and pseudo-labeled unseen
        t_best_val_accuracy, t_best_prompt = self.train_teacher(train_data,
                                                                train_unseen_dataset, 
                                                                val_data)
        # 3. Get teacher pseudo-labels
        log.info(f"Num unlabeled data: {len(unlabeled_examples)}")
        # Decide to finetune either the old samples or new top data
        if config.ALL_UNLABELED:
            std_teacher_preds = self.test_predictions(unlabeled_examples, 
                                                    standard_zsl=True)
        else:
            std_teacher_preds = self.test_predictions(unlabeled_examples, 
                                                      standard_zsl=True) # Check for shape of train_unseen_dataset
            pass
        
        # 4. Take top-16 pseudo-labels to finetune the student
        # Decide to finetune either the old samples or new top data

        self.student = ImagePrefixModel(self.teacher.prefix,
                                        self.teacher.image_pos_emb,
                                        self.image_encoder,
                                        device=self.device).to(self.device)

        return t_best_val_accuracy, t_best_prompt
    
        
    def train_teacher(self, train_data, train_unseen_dataset, val_data):
        # Define dataset for self-training
        unseen_imgs = train_unseen_dataset.filepaths
        unseen_labs = train_unseen_dataset.labels
        seen_imgs = train_data.filepaths
        seen_labs = [self.label_to_idx[l] for l in train_data.labels]

        train_data.filepaths = list(unseen_imgs) + list(seen_imgs)
        train_data.labels = list(unseen_labs) + list(seen_labs)
        
        # Declare the data pre processing for train and validation data
        train_data.label_id = True
        train_data.transform = self.transform
        val_data.transform = self.transform

        train_loader = torch.utils.data.DataLoader(train_data,
                                                   batch_size=self.config.BATCH_SIZE,
                                                   shuffle=True, worker_init_fn=seed_worker,
                                                   generator=g)
        log.info(f"Size of training data: {len(train_data)}")
        # At this time the validation is composed only of seen classes. We can
        # try to expand it with pseudo-labels.
        val_loader = torch.utils.data.DataLoader(val_data,
                                                 batch_size=self.config.BATCH_SIZE)

        accelerator.wait_for_everyone()
        # Load data on accelerate
        self.teacher, self.teacher_optimizer, \
        train_loader, val_loader = accelerator.prepare(self.teacher, 
                                                       self.teacher_optimizer, 
                                                       train_loader,
                                                       val_loader)
        best_val_accuracy = 0
        best_prompt = None
        loss = None

        if val_loader is not None:
            log.info(f"Size of validation data: {len(val_data.filepaths)}")
        # Start teacher training
        for epoch in range(self.config.EPOCHS):
            log.info(f"Run Epoch {epoch}")
            total_loss = 0
            accum_iter = self.config.ACCUMULATION_ITER  
            
            loss, total_loss, epoch_parameters, accuracy = self._train_epoch_teacher(loss, total_loss, 
                                                                                     train_loader, 
                                                                                     accum_iter, epoch)
            accelerator.wait_for_everyone()
            if accelerator.is_local_main_process:
                log.info(f"Loss Epoch {epoch}: {total_loss/(len(train_loader))}")
                log.info(F"Training accuracy after Epoch {epoch}: {accuracy}")
            
            accelerator.free_memory()
            
            if val_loader is not None:
                val_accuracy = self._run_validation(val_loader, epoch)
                log.info(f"Validation accuracy after Epoch {epoch}: {val_accuracy}")
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    best_prompt = epoch_parameters
            else:
                best_val_accuracy = None
                best_prompt = epoch_parameters
        
        return best_val_accuracy, best_prompt

    def _train_epoch_teacher(self, loss, total_loss, train_loader, accum_iter, epoch):

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
            image_features = self.teacher(img)
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
            loss = self.teacher_loss_func(logits, labs)
            total_loss += loss.item()
            
            accelerator.wait_for_everyone()
            
            loss = loss / accum_iter 
            accelerator.backward(loss)

            # Accumulate grandient
            if ((i + 1) % accum_iter == 0) or (i + 1 == len(train_loader)):
                self.teacher_optimizer.step()
                self.teacher.zero_grad()

        accelerator.wait_for_everyone()

        predictions_outputs = accelerator.gather(predictions)
        labels_outputs = accelerator.gather(labels)
        image_outputs = accelerator.gather(images)

        accuracy = np.sum(np.array(predictions_outputs) == np.array(labels_outputs))/len(predictions_outputs)

        current_lr = self.teacher_scheduler.get_last_lr()
        self.teacher_scheduler.step()

        unwrapped_model = accelerator.unwrap_model(self.teacher)
        epoch_parameters = [unwrapped_model.prefix.detach().cpu().numpy(), 
                            unwrapped_model.image_pos_emb.detach().cpu().numpy()]

        return loss, total_loss, epoch_parameters, accuracy

    def _run_validation(self, val_loader, epoch):

        # Define text queries
        prompts = [f"{self.template}{' '.join(i.split('_'))}" \
                        for i in self.seen_classes]
        log.info(f"Number of prompts used for validation: {len(prompts)}")
        # Encode text
        with torch.no_grad():
            text = clip.tokenize(prompts).to(self.device)
            text_features = self.clip_model.encode_text(text)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        predictions = []
        labels = []
        for img, _, _, label, img_path in tqdm(val_loader):
            image_features = self.teacher(img)
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
        

        return accuracy

    def test_predictions(self, data, standard_zsl=False, pseudo=False):
        
        # Declare the data pre processing
        data.transform = self.transform
        # Define the data loader
        test_loader = torch.utils.data.DataLoader(data, 
                                                  batch_size=self.config.BATCH_SIZE)

        accelerator.wait_for_everyone()

        self.teacher, test_loader = accelerator.prepare(self.teacher, test_loader)

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
                image_features = self.teacher(img)
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

