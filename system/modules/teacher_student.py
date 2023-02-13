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
from ..models import CustomImageEncoder, make_scheduler, ImagePrefixModel
from ..modules import VPTPseudoBaseline

g = torch.Generator()
g.manual_seed(0)

log = logging.getLogger(__name__)


class TeacherStudent(VPTPseudoBaseline):
    def __init__(self, config, label_to_idx, data_folder, 
                 classes, seen_classes, unseen_classes,
                 device):
        
        super().__init__(config, label_to_idx, classes, 
                         seen_classes, unseen_classes,
                         device) 

        self.data_folder = data_folder
        
    def train(self, train_data, val_data, unlabeled_data):

        # Number of total iterations to cover all unlabeled data
        num_iter = int(len(unlabeled_data) / self.config.STEP_QUANTILE)
        # Initialize the number of pseudo-labels per class
        self.num_pseudo_labels_per_class = int(num_iter / len(self.unseen_classes))
        log.info(f"We select {self.num_pseudo_labels_per_class} per each unseen classes.")
        # Create a safe copy of labeled/unlabeled data
        original_train_data = copy.deepcopy(train_data)
        original_unlabeled_data = copy.deepcopy(unlabeled_data)
        
        for niter in range(1, num_iter):
            log.info(f"Start first round of training..")
            # Create labeled seen dataset to train on so that it is balanced with unseen classes
            np.random.seed(self.config.validation_seed)
            desired_labeled_data = self.num_pseudo_labels_per_class*len(self.seen_classes)
            # Avoid the error of too many data to samples
            num_labels = min(desired_labeled_data, len(original_train_data.filepaths))
            train_indices = np.random.choice(range(len(original_train_data.filepaths)),
                                            size=num_labels,
                                            replace=False)
            # Update the training data
            train_data.filepaths = [f for i, f in enumerate(original_train_data.filepaths) if i in train_indices]
            train_data.labels = [l for i, l in enumerate(original_train_data.labels) if i in train_indices]

            
            
            # 1. Initialize teacher
            self.define_model(teacher=True)
            log.info(f"[TEACHER] Initialization..")

            # 2. Train teacher with labeled seen and pseudo-labeled unseen
            log.info(f"[TEACHER] Start model training..")
            t_best_val_accuracy, t_best_prompt = self.train_teacher(train_data,
                                                                    val_data,
                                                                    unlabeled_data)
            log.info(f"[TEACHER] Training completed.")

            # 3. Get teacher pseudo-labels
            log.info(f"[TEACHER] Collecting teacher pseudo-labels on unlabeled data..")
            pseudo_labels = self.get_pseudo_labels(original_unlabeled_data,
                                                   teacher=True)

            # 4. Initialize student model
            log.info(f"[STUDENT] Initialization..")
            self.define_model(teacher=False)

            # 5. Train student 
            log.info(f"[STUDENT] Start model training..")
            self.train_student(pseudo_labels)
            log.info(f"[STUDENT] Training completed.")

            # 6. Get new pseudo labels from student
            log.info(f"[STUDENT] Get student pseudo-labels for the next round of training.")
            self.num_pseudo_labels_per_class = int((niter + 1) * (len(original_unlabeled_data) / self.config.STEP_QUANTILE) / len(self.unseen_classes))
            unlabeled_data = self.get_pseudo_labels(original_unlabeled_data,
                                                    teacher=False)

        return t_best_val_accuracy, t_best_prompt

    def train_student(self, train_data):

        train_loader = torch.utils.data.DataLoader(train_data,
                                                   batch_size=self.config.BATCH_SIZE,
                                                   shuffle=True, worker_init_fn=seed_worker,
                                                   generator=g)
        log.info(f"[STUDENT] The size of training data is {len(train_data)}")
        accelerator.wait_for_everyone()
        
        # Load data on accelerate
        self.student, self.student_optimizer, \
        train_loader = accelerator.prepare(self.student, 
                                           self.student_optimizer, 
                                           train_loader)

        loss = None
        # Start teacher training
        for epoch in range(self.config.EPOCHS):
            log.info(f"Run Epoch {epoch}")
            total_loss = 0
            accum_iter = self.config.ACCUMULATION_ITER  
            
            loss, total_loss, accuracy = self._train_epoch(loss, total_loss, 
                                                           train_loader, 
                                                           accum_iter, epoch,
                                                           only_unlabelled=True,
                                                           teacher=False)
            accelerator.wait_for_everyone()
            if accelerator.is_local_main_process:
                log.info(f"Loss Epoch {epoch}: {total_loss/(len(train_loader))}")
                log.info(F"Training accuracy after Epoch {epoch}: {accuracy}")
            
            accelerator.free_memory()
     
    def train_teacher(self, train_data, val_data, unlabeled_data):
        """ This function defines the training of self.model.

        :param train_data: Dataset object - training dataset of labeled data for 
                           seen classes (defined in zsl_jpl line 323)
        :param unlabeled_data: Dataset object - dataset of unlabeled data for 
                               unseen classes (defined in zsl_jpl line 328)
        :param val_data: Dataset object - validation dataset of labeled data for
                         seen classes (defined in zsl_jpl line 334)
        """

        # Define training dataset
        self.create_training_dataset(train_data, unlabeled_data)
        
        # Declare the data pre processing for train and validation data
        train_data.transform = self.transform
        val_data.transform = self.transform

        train_loader = torch.utils.data.DataLoader(train_data,
                                                   batch_size=self.config.BATCH_SIZE,
                                                   shuffle=True, worker_init_fn=seed_worker,
                                                   generator=g)
        # At this time the validation is composed only of seen classes. We can
        # try to expand it with pseudo-labels.
        val_loader = torch.utils.data.DataLoader(val_data,
                                                 batch_size=self.config.BATCH_SIZE)
        log.info(f"[TEACHER] The size of training data: {len(train_data)}")
        
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
            
            loss, total_loss, epoch_parameters = self._train_epoch(loss, total_loss, 
                                                                   train_loader, 
                                                                   accum_iter, epoch,
                                                                   only_unlabelled=False,
                                                                   teacher=True)
            accelerator.wait_for_everyone()
            if accelerator.is_local_main_process:
                log.info(f"Loss Epoch {epoch}: {total_loss/(len(train_loader))}")
                #log.info(F"Training accuracy after Epoch {epoch}: {accuracy}")
            
            accelerator.free_memory()
            
            if val_loader is not None:
                val_accuracy = self._run_validation(val_loader, epoch, teacher=True)
                log.info(f"Validation accuracy after Epoch {epoch}: {val_accuracy}")
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    best_prompt = epoch_parameters
            else:
                best_val_accuracy = None
                best_prompt = epoch_parameters
        
        return best_val_accuracy, best_prompt

    def test_predictions(self, data, standard_zsl=False, 
                         teacher=True):
        
        # Declare the data pre processing
        data.transform = self.transform
        # Define the data loader
        test_loader = torch.utils.data.DataLoader(data, 
                                                  batch_size=self.config.BATCH_SIZE)

        accelerator.wait_for_everyone()

        if teacher:
            self.teacher, test_loader = accelerator.prepare(self.teacher, test_loader)
        else:
            self.student, test_loader = accelerator.prepare(self.student, test_loader)

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
                if teacher:
                    image_features = self.teacher(img)
                else:
                    image_features = self.student(img)
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

    def define_model(self, teacher=True):
            """ This function allows to define the model and its
            - optimizer
            - schedule
            - loss function """

            # Define models
            if teacher:
                self.teacher = ImagePrefixModel(self.vis_initial_prefix,
                                            self.initial_pos_emb,
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
            
            else:
                self.student = ImagePrefixModel(self.teacher.prefix,
                                            self.teacher.image_pos_emb,
                                            self.image_encoder,
                                            device=self.device).to(self.device)
            
                for i, parameter in enumerate(self.student.parameters()):
                    if parameter.requires_grad:
                        log.info(f"Shape of teacher parameters {i}: {parameter.shape}")

                if self.config.t_OPTIM == 'SGD':
                    self.student_optimizer = torch.optim.SGD(self.student.parameters(), 
                                                            lr=self.config.s_LR, 
                                                            weight_decay=self.config.s_DECAY,
                                                            momentum=0.9)
                self.student_scheduler = make_scheduler(self.student_optimizer, self.config)
                self.student_loss_func = torch.nn.CrossEntropyLoss() 

    def define_loss_function(self, logits, labs, teacher=False):
        
        if teacher:
            return self.teacher_loss_func(logits, labs)
        else:
            return self.student_loss_func(logits, labs)

    def training_model(self, img, teacher=False):
        """ This function allows to customize the model to use while trainig

        :param img: Tensor of images form Dataloader
        """
        if teacher:
            return self.teacher(img)
        else:
            return self.student(img)

    def backpropagate(self, teacher=False):

        if teacher:
            self.teacher_optimizer.step()
            self.teacher.zero_grad()
        else:
            self.student_optimizer.step()
            self.student.zero_grad()

    def update_scheduler(self, teacher=False):

        if teacher:
            current_lr = self.teacher_scheduler.get_last_lr()
            self.teacher_scheduler.step()
        else:
            current_lr = self.student_scheduler.get_last_lr()
            self.student_scheduler.step()

    def unwrap_model(self, teacher=False):

        if teacher:
            return accelerator.unwrap_model(self.teacher)
        else:
            return accelerator.unwrap_model(self.student)      

    def get_pseudo_labels(self, unlabeled_examples, teacher=True):

        log.info(f"Num unlabeled data: {len(unlabeled_examples)}")
        # Get prediction of teacher on unlabeled data
        std_preds = self.test_predictions(unlabeled_examples, 
                                          standard_zsl=True,
                                          teacher=teacher)
        
        # 4. Take top-16 pseudo-labels to finetune the student
        pseudo_unseen_examples = CustomDataset(std_preds['id'], 
                                               self.data_folder, 
                                               transform=self.transform, 
                                               augmentations=None, 
                                               train=True, labels=None,
                                               label_map=self.label_to_idx)

        pseudo_labels = self.assign_pseudo_labels(self.num_pseudo_labels_per_class, 
                                                  pseudo_unseen_examples)

        return pseudo_labels

    def assign_pseudo_labels(self, k, unlabeled_data, teacher=True):

        # Define text queries
        prompts = [f"{self.template}{' '.join(i.split('_'))}" \
                    for i in self.unseen_classes]
        log.info(f"Number of prompts: {len(prompts)}")

        # Encode text
        text = clip.tokenize(prompts).to(self.device)
        text_features = self.clip_model.encode_text(text)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # to find the top k for each class, each class has it's own "leaderboard"
        top_k_leaderboard = {self.label_to_idx[self.unseen_classes[i]] : [] 
                            for i in range(len(self.unseen_classes))} #maps class idx -> (confidence, image_path) tuple
    
        for img_path in tqdm(unlabeled_data.filepaths):
            img = Image.open(img_path).convert('RGB')
            img = torch.unsqueeze(self.transform(img), 0).to(self.device)
            with torch.no_grad():
                if teacher:
                    image_features = self.teacher(img)
                else:
                    image_features = self.student(img)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                # cosine similarity as logits        
            
            logit_scale = self.clip_model.logit_scale.exp()
            logits = logit_scale * image_features @ text_features.t()
            probs = logits.softmax(dim=-1)
            idx_preds = torch.argmax(logits, dim=1)
            pred_id = idx_preds.item()
            pred = self.label_to_idx[self.unseen_classes[idx_preds.item()]]

            """if predicted class has empty leaderboard, or if the confidence is high
            enough for predicted class leaderboard, add the new example
            """
            prob_score = probs[0][pred_id]
            if len(top_k_leaderboard[pred]) < k:
                top_k_leaderboard[pred].append((prob_score, img_path))
            elif top_k_leaderboard[pred][-1][0] < prob_score: #if the confidence in predicted class "qualifies" for top-k
                # default sorting of tuples is by first element
                top_k_leaderboard[pred] = sorted(top_k_leaderboard[pred] + [(probs[0][pred_id], img_path)], reverse=True)[:k]
            else:
                #sort the other classes by confidence score
                order_of_classes = sorted([(probs[0][j], j) for j in range(len(self.unseen_classes)) if j != pred_id], reverse=True)
                for score, index in order_of_classes:
                    index_dict = self.label_to_idx[self.unseen_classes[index]]
                    #log.info(f"{classnames[index]}")
                    #log.info(f"{index_dict}")
                    if len(top_k_leaderboard[index_dict]) < k:
                        top_k_leaderboard[index_dict].append((probs[0][index], img_path))
                    elif top_k_leaderboard[index_dict][-1][0] < probs[0][index]:
                        #default sorting of tuples is by first element
                        top_k_leaderboard[index_dict] = sorted(top_k_leaderboard[index_dict] + [((probs[0][index], img_path))], reverse=True)[:k]
        
        new_imgs = []
        new_labels = []
        #loop through, and rebuild the dataset
        for index, leaderboard in top_k_leaderboard.items():
            new_imgs += [tup[1] for tup in leaderboard]
            new_labels += [index for _ in leaderboard]
        
        unlabeled_data.filepaths = new_imgs
        unlabeled_data.labels = new_labels
        unlabeled_data.label_id = True

        return unlabeled_data