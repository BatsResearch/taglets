import os
import numpy as np

from .dataset_api import DatasetAPI

from taglets.data import CustomImageDataset


class FMD(DatasetAPI):
    def __init__(self, dataset_dir, seed=0):
        super().__init__(os.path.join(dataset_dir, 'image'), seed)
        
        self.checkpoint_shot = [1, 5, 20, 50]
        self.classes = np.asarray(['fabric', 'foliage', 'glass', 'leather', 'metal', 'paper', 'plastic', 'stone',
                                   'water', 'wood'])
        self.img_paths = []
        self.labels = []
        for idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.dataset_dir, class_name)
            for img in os.listdir(class_dir):
                if not img.endswith('.jpg'):
                    continue
                self.img_paths.append(os.path.join(class_dir, img))
                self.labels.append(idx)
        self.img_paths = np.asarray(self.img_paths)
        self.labels = np.asarray(self.labels)

        self._init_random()
        
        self.test_indices = np.zeros(500, dtype=np.int)
        for i in range(10):
            self.test_indices[50*i:50*(i+1)] = np.random.choice(100, 50, replace=False) + (100 * i)
        self.train_indices = []
        for i in range(1000):
            if i not in self.test_indices:
                self.train_indices.append(i)
        self.train_indices = np.asarray(self.train_indices)
        
        # shuffle for checkpoints
        np.random.shuffle(self.train_indices)
        
    def get_num_checkpoints(self):
        return len(self.checkpoint_shot)
    
    def get_class_names(self):
        return self.classes
    
    def get_labeled_dataset(self, checkpoint_num):
        shot = self.checkpoint_shot[checkpoint_num]
        if checkpoint_num <= 2:
            checkpoint_indices = self.train_indices[:shot]
            labeled_dataset = CustomImageDataset(self.img_paths[checkpoint_indices],
                                                 labels=self.labels[checkpoint_indices],
                                                 transform=self._get_transform_image(train=True))
            return labeled_dataset, None
        else:
            train_checkpoint_indices = self.train_indices[:int(shot*0.8)]
            val_checkpoint_indices = self.train_indices[int(shot * 0.8):shot]
            labeled_dataset = CustomImageDataset(self.img_paths[train_checkpoint_indices],
                                                 labels=self.labels[train_checkpoint_indices],
                                                 transform=self._get_transform_image(train=True))
            val_dataset = CustomImageDataset(self.img_paths[val_checkpoint_indices],
                                             labels=self.labels[val_checkpoint_indices],
                                             transform=self._get_transform_image(train=False))
            return labeled_dataset, val_dataset
            
    def get_unlabeled_dataset(self, checkpoint_num, train):
        shot = self.checkpoint_shot[checkpoint_num]
        checkpoint_indices = self.train_indices[shot:]
        unlabeled_dataset = CustomImageDataset(self.img_paths[checkpoint_indices],
                                               transform=self._get_transform_image(train=train))
        return unlabeled_dataset
    
    def get_test_dataset(self):
        test_dataset = CustomImageDataset(self.img_paths[self.test_indices],
                                          transform=self._get_transform_image(train=False))
        return test_dataset
    
    def get_test_labels(self):
        return self.labels[self.test_indices]