from pathlib import Path
from random import sample
import os
import numpy as np
import torch
from torchvision import models
from PIL import Image


class ActiveLearningModule:
    """
    The base class for an active learning module. Used to find examples to label.
    """

    def __init__(self, task):
        """
        Create a new ActiveLearningModule.
        :param task: The current task
        """
        self.task = task

    def find_candidates(self, available_budget):
        """
        Find candidates to label.
        :param available_budget: The number of candidates to label
        :return: A list of the filenames of candidates to label
        """
        raise NotImplementedError


class LeastConfidenceActiveLearning(ActiveLearningModule):
    """
    An active learning module that chooses candidates to label based on confidence scores of examples.
    """

    def __init__(self, task):
        """
        Create a new LeastConfidenceActiveLearning module.
        :param task: The current task
        """
        super().__init__(task)
        images, images_names = self.read_images()
        self.image_dict = {}
        for i in range(len(images_names)):
            self.image_dict[images_names[i]] = images[i]

        labeled_images_names, _ = zip(*self.task.labeled_images)
        labeled_images_names = np.asarray(labeled_images_names)
        self.unlabeled_images_names = np.delete(images_names, np.argwhere(np.isin(images_names, labeled_images_names)))
        
        # ---!! TEST ONLY !!!---
        self.unlabeled_images_names = self.unlabeled_images_names[:200]
        # ----------------------
        
        self.model = models.resnet18(pretrained=True)
        self.model.fc = torch.nn.Linear(512, 10, bias=True)
        self.batch_size = 64
        
    def read_images(self):
        """
        read and normalize images
        :return: 4-d tensor and ndarray of str, the images and their names
        """
        image_dir = self.task.unlabeled_image_path
        
        imagenet_mean = np.asarray([0.485, 0.456, 0.406])
        imagenet_std = np.asarray([0.229, 0.224, 0.225])
    
        list_imgs = []
        list_imgs_names = []
        for filename in os.listdir(image_dir):
            if filename.endswith('.png'):
                list_imgs_names.append(filename)
            
                img = Image.open(os.path.join(image_dir, filename))
                rgbimg = img.convert('RGB')
                rgbimg = np.asarray(rgbimg) / 255.0
                normalized_rgbimg = (rgbimg - imagenet_mean) / imagenet_std
                list_imgs.append(torch.from_numpy(normalized_rgbimg.astype(np.float32)))
        list_imgs = torch.stack(list_imgs)
        list_imgs = list_imgs.permute(0, 3, 1, 2)
        return list_imgs, np.asarray(list_imgs_names)
    
    def fine_tune_on_labeled_images(self, use_gpu, num_epochs=1, lr=1e-3):
        """
        !!!!!!!!! num_epochs = 1 is only for testing !!!!!!!!!
        Fine tune the pre-trained model with labeled images in self.task.
        Currently, this function is only called by find_candidates.
        :param: use_gpu: Whether or not to use the GPU
        :param num_epochs: int, number of epochs to fine tune
        :param lr: float, learning rate
        :return:
        """
        self.model.train()
        if use_gpu:
            self.model.cuda()
        else:
            self.model.cpu()

        labeled_images_names, labels = zip(*self.task.labeled_images)
        labeled_images_names = np.asarray(labeled_images_names)
        labels = torch.from_numpy(np.asarray(labels).astype(np.int64))

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        classification_criterion = torch.nn.CrossEntropyLoss()
    
        num_len = labeled_images_names.shape[0]
        for ep in range(num_epochs):
            perm = torch.randperm(num_len)
            for i in range(0, num_len, self.batch_size):
                optimizer.zero_grad()
                
                ind = perm[i: i + self.batch_size]
                batch_images_names = labeled_images_names[ind]
                batch_images = [self.image_dict[image_name] for image_name in batch_images_names]
                batch_images = torch.stack(batch_images)
                batch_labels = labels[ind]
            
                if use_gpu:
                    batch_images = batch_images.cuda()
                    batch_labels = batch_labels.cuda()
            
                # upsample here
                logits = self.model(torch.nn.functional.interpolate(batch_images, (224, 224)))
                loss = classification_criterion(logits, batch_labels)
    
                loss.backward()
                optimizer.step()

    def find_candidates(self, available_budget, use_gpu=False):
        """
        Find candidates to label based on confidence.
        :param available_budget: The number of candidates to label
        :param use_gpu: Whether or not to use the GPU
        :return: A list of the filenames of candidates to label
        """
        # fine tune the pre-trained model
        self.fine_tune_on_labeled_images(use_gpu)
        self.model.eval()
        if use_gpu:
            self.model.cuda()
        else:
            self.model.cpu()
        
        num_len = self.unlabeled_images_names.shape[0]
        list_logits = []
        for i in range(0, num_len, self.batch_size):
            batch_images = [self.image_dict[self.unlabeled_images_names[j]]
                            for j in range(i, min(i + self.batch_size, num_len))]
            batch_images = torch.stack(batch_images)
            
            if use_gpu:
                batch_images = batch_images.cuda()
    
            # upsample here
            logits = self.model(torch.nn.functional.interpolate(batch_images, (224, 224)))
            list_logits.append(logits.cpu().detach().numpy())

        all_logits = np.concatenate(list_logits)
        confidence = np.max(all_logits, axis=1)
        
        least_confidence_indices = np.argsort(confidence)[:available_budget]
        rest_indices = np.argsort(confidence)[available_budget:]
        
        to_request = self.unlabeled_images_names[least_confidence_indices]
        
        # update self.unlabeled_images_names so that we don't request the same images
        self.unlabeled_images_names = self.unlabeled_images_names[rest_indices]
        
        return to_request
        

class RandomActiveLearning(ActiveLearningModule):
    """
    An active learning module that randomly chooses candidates to label.
    """

    def __init__(self, task):
        """
        Create a new RandomActiveLearning module.
        :param task: The current task
        """
        super().__init__(task)
        self.labeled = set()  # List of candidates already labeled

    def find_candidates(self, available_budget):
        """
        Randomly find candidates to label.
        :param available_budget: The number of candidates to label
        :return: A list of the filenames of candidates to label
        """
        image_dir = self.task.unlabeled_image_path
        unlabeled_images = [f.name for f in Path(image_dir).iterdir() if f.is_file() and f.name not in self.labeled]
        to_request = sample(unlabeled_images, min(len(unlabeled_images), available_budget))
        self.labeled.update(to_request)
        return to_request
