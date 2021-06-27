import numpy as np
import random
import torchvision.transforms as transforms


class DatasetAPI:
    def __init__(self, dataset_dir, seed=0):
        self.dataset_dir = dataset_dir
        self.seed = seed
    
    def _get_transform_image(self, train=True):
        """
        Get the transform to be used on an image.
        :return: A transform
        """
        data_mean = [0.485, 0.456, 0.406]
        data_std = [0.229, 0.224, 0.225]
        # Remember to check it for video and eval
        if train:
            return transforms.Compose([
                transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=data_mean, std=data_std)
            ])
        else:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=data_mean, std=data_std)
            ])
        
    def _init_random(self):
        np.random.seed(self.seed)
        random.seed(self.seed)
    
    def get_num_checkpoints(self):
        raise NotImplementedError
    
    def get_class_names(self):
        raise NotImplementedError
    
    def get_labeled_dataset(self, checkpoint_num):
        raise NotImplementedError
        
    def get_unlabeled_dataset(self, checkpoint_num):
        raise NotImplementedError
        
    def get_test_dataset(self):
        raise NotImplementedError
        
    def get_test_labels(self):
        raise NotImplementedError