import os
import numpy as np

from .dataset_api import DatasetAPI

from taglets.data import CustomImageDataset


class Places(DatasetAPI):
    def __init__(self, dataset_dir, seed=0):
        super().__init__(dataset_dir, seed)
        
        self.checkpoint_shot = [1, 5, 20, 50, 100, 500]
        self.classes = np.asarray(os.listdir(os.path.join(dataset_dir, 'train')))
        
        self.train_img_paths, self.train_labels = self._get_images_and_labels('train')
        self.val_img_paths, self.val_labels = self._get_images_and_labels('val')
        
        self._init_random()
        
        self.train_indices = np.arange(len(self.train_labels))
        
        # shuffle for checkpoints
        np.random.shuffle(self.train_indices)
    
    def _get_images_and_labels(self, split):
        img_paths = []
        labels = []
        for idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.dataset_dir, split, class_name)
            for img in os.listdir(class_dir):
                if not img.endswith('.jpg'):
                    continue
                img_paths.append(os.path.join(class_dir, img))
                labels.append(idx)
        img_paths = np.asarray(img_paths)
        labels = np.asarray(labels)
        return img_paths, labels
    
    def get_num_checkpoints(self):
        return len(self.checkpoint_shot)
    
    def get_class_names(self):
        return self.classes
    
    def get_labeled_dataset(self, checkpoint_num):
        shot = self.checkpoint_shot[checkpoint_num]
        if checkpoint_num <= 2:
            checkpoint_indices = self.train_indices[:shot]
            labeled_dataset = CustomImageDataset(self.train_img_paths[checkpoint_indices],
                                                 labels=self.train_labels[checkpoint_indices],
                                                 transform=self._get_transform_image(train=True))
            return labeled_dataset, None
        else:
            train_checkpoint_indices = self.train_indices[:int(shot * 0.8)]
            val_checkpoint_indices = self.train_indices[int(shot * 0.8):shot]
            labeled_dataset = CustomImageDataset(self.train_img_paths[train_checkpoint_indices],
                                                 labels=self.train_labels[train_checkpoint_indices],
                                                 transform=self._get_transform_image(train=True))
            val_dataset = CustomImageDataset(self.train_img_paths[val_checkpoint_indices],
                                             labels=self.train_labels[val_checkpoint_indices],
                                             transform=self._get_transform_image(train=False))
            return labeled_dataset, val_dataset
    
    def get_unlabeled_dataset(self, checkpoint_num, train):
        shot = self.checkpoint_shot[checkpoint_num]
        checkpoint_indices = self.train_indices[shot:]
        unlabeled_dataset = CustomImageDataset(self.train_img_paths[checkpoint_indices],
                                               transform=self._get_transform_image(train=train))
        return unlabeled_dataset
    
    def get_test_dataset(self):
        test_dataset = CustomImageDataset(self.val_img_paths,
                                          transform=self._get_transform_image(train=False))
        return test_dataset
    
    def get_test_labels(self):
        return self.val_labels