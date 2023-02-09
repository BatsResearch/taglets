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

        unseen_to_idx = {c:idx for idx, c in enumerate(self.unseen_classes)}
        self.unseen_idx_to_real = {unseen_to_idx[c]:self.label_to_idx[c] for c in self.unseen_classes}
        self.unseen_real_to_idx = {self.label_to_idx[c]:unseen_to_idx[c] for c in self.unseen_classes}

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
        self.vis_initial_prefix, self.initial_pos_emb = self.initialize_prompts(visual_transformer)
        
    def initialize_teacher(self):
        # Define models
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

        # Number of total iterations to cover all unlabeled data
        num_iter = int(len(unlabeled_data) / self.config.STEP_QUANTILE)
        # Initialize the number of pseudo-labels per class
        num_pseudo_labels_per_class = int((len(unlabeled_data) / self.config.STEP_QUANTILE) / len(self.unseen_classes))
        log.info(f"We select {num_pseudo_labels_per_class} per each unseen classes.")
        # Create a safe copy of labeled/unlabeled data
        original_train_data = copy.deepcopy(train_data)
        original_unlabeled_data = copy.deepcopy(unlabeled_data)
        
        for niter in range(1, num_iter):
            log.info(f"Start first round of training..")
            # Create labeled seen dataset to train on so that it is balanced with unseen classes
            np.random.seed(self.config.validation_seed)
            desired_labeled_data = num_pseudo_labels_per_class*len(self.seen_classes)
            # Avoid the error of too many data to samples
            num_labels = min(desired_labeled_data, len(original_train_data.filepaths))
            train_indices = np.random.choice(range(len(original_train_data.filepaths)),
                                            size=num_labels,
                                            replace=False)
            # Update the training data
            train_data.filepaths = [f for i, f in enumerate(original_train_data.filepaths) if i in train_indices]
            train_data.labels = [l for i, l in enumerate(original_train_data.labels) if i in train_indices]

            # Get pseudo-labels for training data of unseen classes
            train_unseen_dataset = pseudolabel_top_k(num_pseudo_labels_per_class,
                                                    self.config.PROMPT_TEMPLATE,
                                                    unlabeled_data,
                                                    self.unseen_classes,
                                                    self.transform,
                                                    self.clip_model,
                                                    self.label_to_idx,
                                                    self.device)
            
            # 1. Initialize teacher
            self.initialize_teacher()
            log.info(f"[TEACHER] Initialization..")

            # 2. Train teacher with labeled seen and pseudo-labeled unseen
            log.info(f"[TEACHER] Start model training..")
            t_best_val_accuracy, t_best_prompt = self.train_teacher(train_data,
                                                                    train_unseen_dataset, 
                                                                    val_data)
            log.info(f"[TEACHER] Training completed.")

            # 3. Get teacher pseudo-labels
            log.info(f"[TEACHER] Collecting teacher pseudo-labels on unlabeled data..")
            pseudo_labels = self.get_pseudo_labels(original_unlabeled_data,
                                                   num_pseudo_labels_per_class,
                                                   teacher=True)

            # 4. Initialize student model
            log.info(f"[STUDENT] Initialization..")
            self.initialize_student()

            # 5. Train student 
            log.info(f"[STUDENT] Start model training..")
            self.train_student(pseudo_labels)
            log.info(f"[STUDENT] Training completed.")

            # 6. Get new pseudo labels from student
            log.info(f"[STUDENT] Get student pseudo-labels for the next round of training.")
            num_pseudo_labels_per_class = int((niter + 1) * (len(original_unlabeled_data) / self.config.STEP_QUANTILE) / len(self.unseen_classes))
            unlabeled_data = self.get_pseudo_labels(original_unlabeled_data,
                                                    num_pseudo_labels_per_class,
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
            
            loss, total_loss, accuracy = self._train_epoch_student(loss, total_loss, 
                                                                    train_loader, 
                                                                    accum_iter, epoch)
            accelerator.wait_for_everyone()
            if accelerator.is_local_main_process:
                log.info(f"Loss Epoch {epoch}: {total_loss/(len(train_loader))}")
                log.info(F"Training accuracy after Epoch {epoch}: {accuracy}")
            
            accelerator.free_memory()

    def _train_epoch_student(self, loss, total_loss, train_loader, accum_iter, epoch):
        # Define text queries
        prompts = [f"{self.template}{' '.join(i.split('_'))}" \
                        for i in self.unseen_classes]
        log.info(f"[STUDENT] Number of prompts: {len(prompts)}")
        # Encode text
        with torch.no_grad():
            text = clip.tokenize(prompts).to(self.device)
            text_features = self.clip_model.encode_text(text)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        predictions = []
        images = []
        labels = []
        for i, (img, _, _, label, img_path) in enumerate(tqdm(train_loader)):
            image_features = self.student(img)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # cosine similarity as logits        
            logit_scale = self.clip_model.logit_scale.exp()
            logits = logit_scale * image_features @ text_features.t()

            idx_preds = torch.argmax(logits, dim=1)
            real_preds = [self.unseen_classes[i.item()] for i in idx_preds]
            
            predictions += real_preds
            labels += [self.classes[i.item()] for i in label]
            images += [i for i in img_path]

            labs = torch.tensor([self.unseen_real_to_idx[l.item()] for l in label])
            loss = self.student_loss_func(logits, labs)
            total_loss += loss.item()
            
            accelerator.wait_for_everyone()
            
            loss = loss / accum_iter 
            accelerator.backward(loss)

            # Accumulate grandient
            if ((i + 1) % accum_iter == 0) or (i + 1 == len(train_loader)):
                self.student_optimizer.step()
                self.student.zero_grad()

        accelerator.wait_for_everyone()

        predictions_outputs = accelerator.gather(predictions)
        labels_outputs = accelerator.gather(labels)
        image_outputs = accelerator.gather(images)

        accuracy = np.sum(np.array(predictions_outputs) == np.array(labels_outputs))/len(predictions_outputs)

        current_lr = self.student_scheduler.get_last_lr()
        self.student_scheduler.step()


        return loss, total_loss, accuracy
        
    def initialize_student(self):

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

    def get_pseudo_labels(self, unlabeled_examples, num_pseudo_labels_per_class, teacher=True):

        log.info(f"Num unlabeled data: {len(unlabeled_examples)}")
        # Decide to finetune either the old samples or new top data
        if self.config.ALL_UNLABELED:
            std_preds = self.test_predictions(unlabeled_examples, 
                                                      standard_zsl=True,
                                                      teacher=teacher)
        else:
            std_preds = self.test_predictions(unlabeled_examples, 
                                                      standard_zsl=True,
                                                      teacher=teacher)
        
        # 4. Take top-16 pseudo-labels to finetune the student
        # Decide to finetune either the old samples or new top data
        pseudo_unseen_examples = CustomDataset(std_preds['id'], 
                                               self.data_folder, 
                                               transform=self.transform, 
                                               augmentations=None, 
                                               train=True, labels=None,
                                               label_map=self.label_to_idx)

        pseudo_labels = self.assign_pseudo_labels(num_pseudo_labels_per_class, 
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
        log.info(f"[TEACHER] The size of training data: {len(train_data)}")
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

    def test_predictions(self, data, standard_zsl=False, pseudo=False, teacher=True):
        
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

