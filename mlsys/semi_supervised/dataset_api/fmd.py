import os
import numpy as np

from .dataset_api import DatasetAPI


class FMD(DatasetAPI):
    def __init__(self, dataset_dir, seed=0):
        super().__init__(dataset_dir, seed)
        
        self.checkpoint_shot = [1, 5, 20, 50]
        self.classes = np.asarray(['fabric', 'foliage', 'glass', 'leather', 'metal', 'paper', 'plastic', 'stone',
                                   'water', 'wood'])
        self.all_img_paths = []
        for class_name in self.classes:
            img_paths = []
            class_dir = os.path.join(self.dataset_dir, 'image', class_name)
            for img in os.listdir(class_dir):
                if not img.endswith('.jpg'):
                    continue
                img_paths.append(os.path.join(class_dir, img))
            self.all_img_paths.append(np.asarray(img_paths))

        self._init_random()
        
        self.test_img_paths = []
        self.test_labels = []
        for i in range(10):
            indices = np.arange(100)
            np.random.shuffle(indices)
            self.test_img_paths.append(self.all_img_paths[i][indices[:20]])
            self.test_labels = self.test_labels + ([i] * 20)
            self.all_img_paths[i] = self.all_img_paths[i][indices[20:]]
        self.test_img_paths = np.concatenate(np.asarray(self.test_img_paths))
        self.test_labels = np.asarray(self.test_labels)
        self.all_img_paths = np.asarray(self.all_img_paths)
        
        self.train_indices = [np.random.permutation(80) for _ in range(10)]
