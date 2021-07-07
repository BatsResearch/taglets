import os
import numpy as np

from .dataset_api import DatasetAPI


class OfficeHome(DatasetAPI):
    def __init__(self, dataset_dir, seed=0):
        super().__init__(dataset_dir, seed)
        
        domain = 'Clipart'
        
        self.checkpoint_shot = [1, 5, 20]

        data_dir = os.path.join(self.dataset_dir, domain)
        
        self.classes = os.listdir(data_dir)
        
        self.all_img_paths = []
        for class_name in self.classes:
            img_paths = []
            class_dir = os.path.join(self.dataset_dir, domain, class_name)
            for img in os.listdir(class_dir):
                if not img.endswith('.jpg'):
                    continue
                img_paths.append(os.path.join(class_dir, img))
            self.all_img_paths.append(np.asarray(img_paths))
        
        # clean up class names
        fix_class_dict = {'postit_notes': 'post_it_notes'}
        for i in range(len(self.classes)):
            self.classes[i] = self.classes[i].lower()
            if self.classes[i] in fix_class_dict:
                self.classes[i] = fix_class_dict[self.classes[i]]
        self.classes = np.asarray(self.classes)

        self._init_random()

        self.test_img_paths = []
        self.test_labels = []
        for i in range(len(self.classes)):
            indices = np.arange(len(self.all_img_paths[i]))
            np.random.shuffle(indices)
            self.test_img_paths.append(self.all_img_paths[i][indices[-10:]])
            self.test_labels = self.test_labels + ([i] * 10)
            self.all_img_paths[i] = self.all_img_paths[i][indices[:-10]]
        self.test_img_paths = np.concatenate(np.asarray(self.test_img_paths))
        self.test_labels = np.asarray(self.test_labels)

        self.train_indices = []
        for i in range(len(self.classes)):
            self.train_indices.append(np.random.permutation(len(self.all_img_paths[i])))
