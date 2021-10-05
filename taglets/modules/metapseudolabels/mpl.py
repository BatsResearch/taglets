import copy
import math
import os
import torch
import logging
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision
from .model import KNOWN_MODELS


from copy import deepcopy
from enum import Enum
from accelerate import Accelerator
accelerator = Accelerator()

from ..module import Module
from ...pipeline import ImageTaglet, Cache
import numpy as np 

log = logging.getLogger(__name__)

from .utils import RandAugTransform, is_grayscale

def softXEnt(input, target):
    return -torch.sum(F.log_softmax(input, dim=1) * target, dim=1)


class MetaPseudoLabelsModule(Module):
    def __init__(self, task):
        super().__init__(task)
        self.taglets = [MetaPseudoLabelsTaglet(task)]


class MetaPseudoLabelsTaglet(ImageTaglet):
    def __init__(self, task):
        super().__init__(task)
        self.name = 'metapseudolabels'

        if os.getenv("LWLL_TA1_PROB_TASK") is not None:
            self.save_dir = os.path.join('/home/tagletuser/trained_models', self.name)
        else:
            self.save_dir = os.path.join('trained_models', self.name)
        if not os.path.exists(self.save_dir) and accelerator.is_local_main_process:
            os.makedirs(self.save_dir, exist_ok=True)
        accelerator.wait_for_everyone()

        num_classes = len(self.task.classes)
        self.teacher = models.resnet18(pretrained=False)
        m = torch.nn.Sequential(*list(self.teacher.children())[:-1])
        output_shape = self._get_model_output_shape(self.task.input_shape, m)
        self.teacher.fc = torch.nn.Linear(output_shape, num_classes)

        #self.teacher = KNOWN_MODELS['BiT-M-R50x1'](head_size=num_classes,
        #                                    zero_head=True)
        #self.teacher.load_from(np.load("BiT-M-R50x1.npz"))
        #self.teacher.head.conv = torch.nn.Identity()
        #self.teacher.fc = torch.nn.Conv2d(2048, num_classes, kernel_size=1, bias=True)
        #with torch.no_grad():
        #    torch.nn.init.zeros_(self.teacher.fc.weight)
        #    torch.nn.init.zeros_(self.teacher.fc.bias)

        self.student = models.resnet18(pretrained=False)
        m = torch.nn.Sequential(*list(self.student.children())[:-1])
        output_shape = self._get_model_output_shape(self.task.input_shape, m)
        self.student.fc = torch.nn.Linear(output_shape, num_classes)

        teacher_params_to_update = []
        for param in self.teacher.parameters():
            if param.requires_grad:
                teacher_params_to_update.append(param)

        student_params_to_update = []
        for param in self.student.parameters():
            if param.requires_grad:
                student_params_to_update.append(param)
        

        self.unlabeled_batch_size = self.batch_size
        
        # nesterov with momentum 
        self.student_lr = 0.0005
        self.teacher_lr = 0.0005

        self.teacher_momentum  = 0.9 
        self.teacher_optimizer = torch.optim.SGD(self.teacher.parameters(), 
                                                 lr=0.003, 
                                                 momentum=self.teacher_momentum) 

        #self.teacher_optimizer = torch.optim.Adam(teacher_params_to_update, lr=0.0005)

        self.student_momentum  = 0.9 
        self.student_optimizer = torch.optim.SGD(self.student.parameters(), 
                                                 lr=0.003, 
                                                 momentum=self.student_momentum) 
        #self.student_optimizer = torch.optim.Adam(student_params_to_update, lr=0.0005)

        self.uda_steps = 200
        self.warmup_steps = 0
        self.wait_steps = 5
        self.total_steps = 400

        self.num_epochs = 100

        # TODO: implement these 
        self.teacher_scheduler = self.get_cosine_schedule_with_warmup(self.teacher_optimizer, 
                                                                      self.warmup_steps,
                                                                      self.total_steps) 
        self.student_scheduler = self.get_cosine_schedule_with_warmup(self.student_optimizer, 
                                                                      self.warmup_steps,
                                                                      self.total_steps,
                                                                      self.wait_steps) 

        #self.student_scheduler = torch.optim.lr_scheduler.StepLR(self.student_optimizer, step_size=10, gamma=0.1)
        #self.teacher_scheduler = torch.optim.lr_scheduler.StepLR(self.teacher_optimizer, step_size=10, gamma=0.1)



        self.use_uda_obj = True 
        self.use_sup_obj = True 

        self.CE = torch.nn.CrossEntropyLoss()

        self.lambda_u = 1
        self.temp = 0.7
        self.thresh = 0.95

        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.1)


        #self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.1)

    def get_cosine_schedule_with_warmup(self, 
                                    optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_wait_steps=0,
                                    num_cycles=0.5,
                                    last_epoch=-1):
        def lr_lambda(current_step):
            if current_step < num_wait_steps:
                return 0.0

            if current_step < num_warmup_steps + num_wait_steps:
                return float(current_step) / float(max(1, num_warmup_steps + num_wait_steps))

            progress = float(current_step - num_warmup_steps - num_wait_steps) / \
                float(max(1, num_training_steps - num_warmup_steps - num_wait_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)
    
    def _train(self, train_data_loader, unlabeled_data_loader):
        labeled_iter = iter(train_data_loader)
        unlabeled_iter = iter(unlabeled_data_loader)

        running_loss = 0.0
        running_acc = 0.0
        acc_count = 0


        self.teacher_optimizer, self.student_optimizer = accelerator.prepare(self.teacher_optimizer, self.student_optimizer)
        self.teacher, self.student = accelerator.prepare(self.teacher, self.student)

        self.teacher.train()
        self.student.train()
        for i in range(self.total_steps):
            log.info(f'Starting step {i} of {self.total_steps}')
            try:
                inputs_x, targets_x = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(train_data_loader)
                inputs_x, targets_x = next(labeled_iter)
            
            try:
                ex = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_data_loader)
                ex = next(unlabeled_iter)

            try:
                inputs_u_w, inputs_u_s = ex[0], ex[1]
            except TypeError:
                if self.use_uda_obj:
                    raise ValueError("Unlabeled transform is not configured correctly.")

            batch_size = inputs_x.shape[0]
            inputs = torch.cat((inputs_x, inputs_u_w, inputs_u_s))
            
            teach_logits = self.teacher(inputs)
            labeled_teach_logits = teach_logits[:batch_size]
            ul_teach_logits_w, ul_teach_logits_s = teach_logits[batch_size:].chunk(2)
            del teach_logits

            teach_loss_label = self.CE(labeled_teach_logits, targets_x)

            soft_label = torch.softmax(ul_teach_logits_w.detach() / self.temp, dim=-1)
            probs, hard_label = torch.max(soft_label, dim=-1)
            mask = probs.ge(self.thresh).float()

            teach_loss_unlabel = torch.mean(
                -(soft_label * torch.log_softmax(ul_teach_logits_s, dim=-1)).sum(dim=-1) * mask
            )

            weight_u = self.lambda_u * min(1.0, (i + 1) / self.uda_steps)
            t_loss_uda = teach_loss_label + weight_u * teach_loss_unlabel

            student_inputs = torch.cat((inputs_x, inputs_u_s))
            student_logits = self.student(student_inputs)
            student_logits_l = student_logits[:batch_size]
            student_logits_us = student_logits[batch_size:]
            del student_logits

            student_loss_l_old = F.cross_entropy(student_logits_l, targets_x)
            # ??? change to weak logits 
            student_loss = self.CE(student_logits_us, hard_label)
            accelerator.backward(student_loss)
            self.student_optimizer.step()
            self.student_scheduler.step()


            with torch.no_grad():
                student_logits_l = self.student(inputs_x)
            student_loss_l_new = F.cross_entropy(student_logits_l, targets_x)

            # TODO: tonight; check this 
            delta = student_loss_l_old - student_loss_l_new
            _, hard_psl = torch.max(ul_teach_logits_s.detach(), dim=-1)

            teach_loss_mpl = delta * F.cross_entropy(ul_teach_logits_s, hard_psl)
            teach_loss = t_loss_uda + teach_loss_mpl

            accelerator.backward(teach_loss)
            self.teacher_optimizer.step()
            self.teacher_scheduler.step()

            self.teacher.zero_grad()
            self.student.zero_grad()
            if i % 100:
                log.info(f'student loss: {student_loss}')
                log.info(f'teacher loss: {teach_loss}')


    def _do_train(self, train_data, val_data, unlabeled_data=None):
        log.info('Beginning training')

        train_data_loader = self._get_dataloader(data=train_data, shuffle=True,
                                                 batch_size=self.batch_size)

        if unlabeled_data is not None:
            self.unlabeled_batch_size = min(self.unlabeled_batch_size, len(unlabeled_data))

            unlabeled_data_cp = deepcopy(unlabeled_data)

            unlabeled_data_cp.transform = RandAugTransform(mean=[0.485, 0.456, 0.406],
                                                                 std=[0.229, 0.224, 0.225],
                                                                 input_shape=self.task.input_shape,
                                                                 grayscale=is_grayscale(
                                                                     unlabeled_data.transform))

            unlabeled_data_loader = accelerator.prepare(torch.utils.data.DataLoader(
                dataset=unlabeled_data_cp, batch_size=self.unlabeled_batch_size, shuffle=True,
                num_workers=self.num_workers, pin_memory=True))
            self._train(train_data_loader, unlabeled_data_loader)
        
        self.optimizer = torch.optim.Adam(self.student.parameters(), lr=0.03)
        self.model = self.student 
        #self.optimizer = self.student_optimizer
        #self.lr_scheduler = None
        super()._do_train(train_data, val_data, unlabeled_data)
        


        #if val_data is None:
        #    val_data_loader = None
        #else:
        #    val_data_loader = self._get_dataloader(data=val_data, shuffle=False)
        
        #accelerator.wait_for_everyone()
        #self._finetune(train_data_loader, val_data_loader)
        #accelerator.wait_for_everyone()get_cosine_schedule_with_warmup

        #self.model = accelerator.unwrap_model(self.model)
        #self.model.cpu()
        accelerator.free_memory()

    def _get_pred_classifier(self):
        return self.model 
