import copy
import math
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
from ..utils import seed_worker, dataset_object, evaluate_predictions
from ..models import CustomImageEncoder, make_scheduler, ImagePrefixModel
from ..modules import VPTPseudoBaseline

g = torch.Generator()
g.manual_seed(0)

log = logging.getLogger(__name__)


class TeacherStudent(VPTPseudoBaseline):
    def __init__(self, config, label_to_idx, data_folder, 
                 classes, seen_classes, unseen_classes,
                 device):
        
        super().__init__(config, label_to_idx,
                         classes, seen_classes, unseen_classes,
                         device) 

        self.data_folder = data_folder

        
    def train(self, train_data, val_data, unlabeled_data, test_data,
              test_labeled_files, test_labeles):

        # Number of total iterations to cover all unlabeled data
        num_iter = self.config.STEP_QUANTILE
        num_samples = int(len(unlabeled_data) / num_iter)
        # Initialize the number of pseudo-labels per class
        n_per_class = int(num_samples / len(self.unseen_classes)) 
        n_unseen = len(self.unseen_classes)
        if n_per_class*n_unseen <= len(unlabeled_data.filepaths):
            # self.num_pseudo_labels_per_class =  n_per_class
            self.config.N_PSEUDOSHOTS = n_per_class
        else:
            # self.num_pseudo_labels_per_class =  math.floor(len(unlabeled_data.filepaths)/n_unseen)
            self.config.N_PSEUDOSHOTS = math.floor(len(unlabeled_data.filepaths)/n_unseen)
        
        log.info(f"We select {self.config.N_PSEUDOSHOTS} per each unseen classes.")
        # Create a safe copy of labeled/unlabeled data
        original_train_data = copy.deepcopy(train_data)
        #log.info(f"Training data labels: {original_train_data.labels}")
        original_unlabeled_data = copy.deepcopy(unlabeled_data)
        # Original val
        original_val_data = copy.deepcopy(val_data)
        
        # Initialize here first batch of pseudo labels
        # Define training dataset
        log.info(f"BEFORE: {unlabeled_data.labels}")
        self.create_training_dataset(train_data, unlabeled_data)
        log.info(f"Labels unlabeled data: {unlabeled_data.labels}")

        for niter in range(1, num_iter): 
            log.info(f"Start {niter} round of training..")

            #if niter > 1:
            # Update the training data
            train_data.filepaths = [f for i, f in enumerate(original_train_data.filepaths)]
            train_data.labels = [l for i, l in enumerate(original_train_data.labels)]
            self.update_training_set(train_data, unlabeled_data)

            
            # 1. Initialize teacher
            self.define_model(teacher=True)
            log.info(f"[TEACHER] Initialization..")


            # At this time the validation is composed only of seen classes. We can
            # try to expand it with pseudo-labels.
            if self.val_unseen_files is not None:
                seen_imgs = original_val_data.filepaths
                seen_labs = [self.label_to_idx[l] for l in original_val_data.labels]

                unseen_imgs = list(self.val_unseen_files)
                unseen_labs = list(self.val_unseen_labs)

                val_data.filepaths = list(unseen_imgs) + list(seen_imgs)
                val_data.labels = list(unseen_labs) + list(seen_labs)
                val_data.label_id = True

            # 2. Train teacher with labeled seen and pseudo-labeled unseen
            log.info(f"[TEACHER] Start model training..")
            t_best_val_accuracy, t_best_prompt = self.train_teacher(train_data,
                                                                    val_data)
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
            if self.config.ALL_UNLABELED:
                n_per_class = int((niter + 1) * num_samples / n_unseen)
                if n_per_class*n_unseen <= len(original_unlabeled_data.filepaths):
                    #self.num_pseudo_labels_per_class =  n_per_class
                    self.config.N_PSEUDOSHOTS =  n_per_class
                else:
                    # self.num_pseudo_labels_per_class =  math.floor(len(original_unlabeled_data.filepaths)/n_unseen)
                    self.config.N_PSEUDOSHOTS =  math.floor(len(original_unlabeled_data.filepaths)/n_unseen)

            unlabeled_data = self.get_pseudo_labels(original_unlabeled_data,
                                                    teacher=False)
            # Evaluate model at this point in time
            std_predictions = self.test_predictions(test_data, 
                                                    standard_zsl=True)

            # Submit predictions (standard)
            std_response = evaluate_predictions(std_predictions, test_labeled_files, test_labeles, 
                                                self.unseen_classes, standard_zsl=True)
            log.info(f"[ITERATION] ZSL accuracy: {std_response}")
            
            # Validate on test set (general)
            gen_predictions = self.test_predictions(test_data, 
                                                     standard_zsl=False)
            # Submit predictions (general)
            unseen_accuracy, seen_accuracy, harmonic_mean = evaluate_predictions(gen_predictions, 
                                                                                test_labeled_files, test_labeles, 
                                                                                self.unseen_classes, self.seen_classes, 
                                                                                standard_zsl=False)
            log.info(f'[ITERATION] Generalized ZSL results')
            log.info(f"[ITERATION] Accuracy seen classes: {seen_accuracy}")
            log.info(f"[ITERATION] Accuracy unseen classes: {unseen_accuracy}")
            log.info(f"[ITERATION] Harmonic mean: {harmonic_mean}")



        return t_best_val_accuracy, t_best_prompt

    def update_training_set(self, train_data, unlabeled_data):
        # Get pseudo-labels for unlabeled data from unseen classes
        train_unseen_dataset = unlabeled_data
        # Define the lists of traiing data from seen and unseen classes
        unseen_imgs = train_unseen_dataset.filepaths
        unseen_labs = train_unseen_dataset.labels

        # Use a portion of the pseudo-labeled data to build a validation set
        if self.config.N_PSEUDOSHOTS >= 10:
            np.random.seed(self.config.validation_seed)
            train_indices = np.random.choice(range(len(unseen_imgs)),
                                    size=int(len(unseen_imgs)*self.config.ratio_train_val),
                                    replace=False)
            val_indices = list(set(range(len(unseen_imgs))).difference(set(train_indices)))

            self.val_unseen_files = np.array(unseen_imgs)[val_indices]
            self.val_unseen_labs = np.array(unseen_labs)[val_indices]

            unseen_imgs = list(np.array(unseen_imgs)[train_indices])
            unseen_labs = list(np.array(unseen_labs)[train_indices])   
        else:
            self.val_unseen_files = None
            self.val_unseen_labs = None

        seen_imgs = train_data.filepaths
        seen_labs = [self.label_to_idx[l] for l in train_data.labels]

        self.balance_param = len(seen_imgs)/len(unseen_imgs)

        train_data.filepaths = list(unseen_imgs) + list(seen_imgs)
        train_data.labels = list(unseen_labs) + list(seen_labs)
        train_data.label_id = True
        log.info(f"UPDATE DATASET: size = {len(train_data.filepaths)}")
        log.info(f"UPDATE UNSEEN DATASET: size = {len(unseen_imgs)}")
        log.info(f"UPDATE SEEN DATASET: size = {len(seen_imgs)}")

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
        for epoch in range(self.config.s_EPOCHS):
            log.info(f"Run Epoch {epoch}")
            total_loss = 0
            accum_iter = self.config.ACCUMULATION_ITER  
            
            loss, total_loss, epoch_param = self._train_epoch(loss, total_loss, 
                                                           train_loader, 
                                                           accum_iter, epoch,
                                                           only_unlabelled=True,
                                                           teacher=False)
            accelerator.wait_for_everyone()
            if accelerator.is_local_main_process:
                log.info(f"Loss Epoch {epoch}: {total_loss/(len(train_loader))}")
            
            accelerator.free_memory()

    def train_teacher(self, train_data, val_data):
        """ This function defines the training of self.model.

        :param train_data: Dataset object - training dataset of labeled data for 
                           seen classes (defined in zsl_jpl line 323)
        :param val_data: Dataset object - validation dataset of labeled data for
                         seen classes (defined in zsl_jpl line 334)
        """

        # Declare the data pre processing for train and validation data
        train_data.transform = self.transform
        val_data.transform = self.transform

        train_loader = torch.utils.data.DataLoader(train_data,
                                                   batch_size=self.config.BATCH_SIZE,
                                                   shuffle=True, worker_init_fn=seed_worker,
                                                   generator=g)
        

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
        for epoch in range(self.config.t_EPOCHS):
            log.info(f"Run Epoch {epoch}")
            total_loss = 0
            accum_iter = self.config.ACCUMULATION_ITER  
            
            loss, total_loss, epoch_parameters = self._train_epoch(loss, total_loss, 
                                                                   train_loader, 
                                                                   accum_iter, epoch,
                                                                   teacher=True)
            accelerator.wait_for_everyone()
            if accelerator.is_local_main_process:
                log.info(f"Loss Epoch {epoch}: {total_loss/(len(train_loader))}")
                #log.info(F"Training accuracy after Epoch {epoch}: {accuracy}")
            
            accelerator.free_memory()
            
            if val_loader is not None:
                val_accuracy = self._run_validation(val_loader, teacher=True)
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
        # This is required for distributed training
        test_files = [f.split('/')[-1] for f in test_loader.dataset.filepaths]

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

        predictions = torch.tensor([self.label_to_idx[p] for p in predictions]).to(self.device)
        images = torch.tensor([test_files.index(img) for img in images]).to(self.device)
        
        accelerator.wait_for_everyone()

        predictions_outputs = accelerator.gather(predictions)
        image_outputs = accelerator.gather(images)

        predictions_outputs = [self.classes[p] for p in predictions_outputs]
        image_outputs = [test_files[i] for i in image_outputs]


        df_predictions = pd.DataFrame({'id': image_outputs, 
                                       'class': predictions_outputs})
        df_predictions.drop_duplicates(subset=['id', 'class'], inplace=True)

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

                self.teacher_scheduler = make_scheduler(self.teacher_optimizer, self.config, True, True)
                self.teacher_loss_func = torch.nn.CrossEntropyLoss()
            
            else:
                self.student = ImagePrefixModel(self.vis_initial_prefix,#self.teacher.prefix,
                                            self.initial_pos_emb,#self.teacher.image_pos_emb,
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
                self.student_scheduler = make_scheduler(self.student_optimizer, self.config, True, False)
                self.student_loss_func = torch.nn.CrossEntropyLoss() 

    def define_loss_function(self, logits, labs, teacher=False):
        
        if teacher:
            loss_ce_seen = self.cross_entropy(logits, labs, self.seen_classes)
            loss_ce_unseen = self.cross_entropy(logits, labs, self.unseen_classes)
            return loss_ce_seen + self.balance_param*loss_ce_unseen
        else:
            return self.student_loss_func(logits, labs)

    def cross_entropy(self, logits, labels, classes):
        """ This loss computes the probability mass on the
        opposite set of classes for each sample.
        
        :param logits: continuous vector
        :param labels: class ids
        """

        ids = [self.label_to_idx[c] for c in classes]

        # Get indices of unseen and seen samples in the batch
        samples = [] 
        
        for idx, l in enumerate(labels):
            if l in ids:
                samples.append(idx)

        # Get logit sums on unseen samples
        if samples:
            error = self.teacher_loss_func(logits[samples], labels[samples]) 
        else:
            error = 0
        
        return error

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
        
        DatasetObject = dataset_object(self.config.DATASET_NAME)
        # 4. Take top-16 pseudo-labels to finetune the student
        pseudo_unseen_examples = DatasetObject(std_preds['id'], 
                                               self.data_folder, 
                                               transform=self.transform, 
                                               augmentations=None, 
                                               train=True, labels=None,
                                               label_map=self.label_to_idx)

        pseudo_labels = self.assign_pseudo_labels(self.config.N_PSEUDOSHOTS, 
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
