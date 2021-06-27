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
        self.all_img_paths = []
        for class_name in self.classes:
            img_paths = []
            class_dir = os.path.join(self.dataset_dir, class_name)
            for img in os.listdir(class_dir):
                if not img.endswith('.jpg'):
                    continue
                img_paths.append(os.path.join(class_dir, img))
            self.all_img_paths.append(img_paths)

        self._init_random()
        
        self.test_indices = np.asarray([np.random.choice(100, 50, replace=False) for _ in range(10)])
        self.train_indices = []
        for i in range(10):
            class_test_indices = []
            for j in range(100):
                if j not in self.test_indices[i]:
                    class_test_indices.append(j)
            class_test_indices = np.asarray(class_test_indices)
            np.random.shuffle(class_test_indices)
            self.train_indices.append(class_test_indices)
        self.train_indices = np.asarray(self.train_indices)
        
    def get_num_checkpoints(self):
        return len(self.checkpoint_shot)
    
    def get_class_names(self):
        return self.classes
    
    def get_labeled_dataset(self, checkpoint_num):
        shot = self.checkpoint_shot[checkpoint_num]
        img_paths = []
        labels = []
        for i in range(10):
            checkpoint_indices = self.train_indices[i][:shot]
            img_paths = img_paths + self.all_img_paths[i][checkpoint_indices]
            labels = labels + ([i] * len(checkpoint_indices))
        img_paths = np.asarray(img_paths)
        labels = np.asarray(labels)
        
        
        if checkpoint_num <= 2:
            labeled_dataset = CustomImageDataset(img_paths,
                                                 labels=labels,
                                                 transform=self._get_transform_image(train=True))
            return labeled_dataset, None
        else:
            indices = list(range(len(labels)))
            train_split = int(np.floor(0.8 * len(labels)))
            np.random.shuffle(indices)
            train_idx = indices[:train_split]
            val_idx = indices[train_split:]
            labeled_dataset = CustomImageDataset(img_paths[train_idx],
                                                 labels=labels[train_idx],
                                                 transform=self._get_transform_image(train=True))
            val_dataset = CustomImageDataset(img_paths[val_idx],
                                             labels=labels[val_idx],
                                             transform=self._get_transform_image(train=False))
            return labeled_dataset, val_dataset
            
    def get_unlabeled_dataset(self, checkpoint_num):
        if checkpoint_num == len(self.checkpoint_shot)-1:
            return None

        shot = self.checkpoint_shot[checkpoint_num]
        img_paths = []
        for i in range(10):
            checkpoint_indices = self.train_indices[i][shot:]
            img_paths = img_paths + self.all_img_paths[i][checkpoint_indices]
        img_paths = np.asarray(img_paths)

        unlabeled_train_dataset = CustomImageDataset(img_paths,
                                                     transform=self._get_transform_image(train=True))
        unlabeled_test_dataset = CustomImageDataset(img_paths,
                                                    transform=self._get_transform_image(train=False))
        return unlabeled_train_dataset, unlabeled_test_dataset
    
    def get_unlabeled_labels(self, checkpoint_num):
        if checkpoint_num == len(self.checkpoint_shot)-1:
            return None

        shot = self.checkpoint_shot[checkpoint_num]
        labels = []
        for i in range(10):
            checkpoint_indices = self.train_indices[i][shot:]
            labels = labels + ([i] * len(checkpoint_indices))
        labels = np.asarray(labels)
        return labels
    
    def get_test_dataset(self):
        img_paths = []
        for i in range(10):
            img_paths = img_paths + self.all_img_paths[i][self.test_indices[i]]
        img_paths = np.asarray(img_paths)
        test_dataset = CustomImageDataset(img_paths,
                                          transform=self._get_transform_image(train=False))
        return test_dataset
    
    def get_test_labels(self):
        labels = []
        for i in range(10):
            labels = labels + ([i] * len(self.test_indices[i]))
        labels = np.asarray(labels)
        return labels