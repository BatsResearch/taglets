import csv
import os
import numpy as np

from .dataset_api import DatasetAPI

from taglets.data import CustomImageDataset


class Places205(DatasetAPI):
    def __init__(self, dataset_dir, seed=0):
        super().__init__(dataset_dir, seed)

        self.checkpoint_shot = [1, 5, 20, 50]

        self.classes = []
        self.all_img_paths = []
        img_paths = []
        data_dir = os.path.join(self.dataset_dir, 'data')
        with open(os.path.join(dataset_dir, 'trainvalsplit', 'train_places205.csv'), 'r') as f:
            data = csv.reader(f, delimiter=' ')
            for row in data:
                if len(self.classes) == int(row[1]):
                    if len(img_paths) != 0:
                        self.all_img_paths.append(np.asarray(img_paths))
                        img_paths = []
            
                    name_split = row[0].split("/")
                    if len(name_split) == 3:
                        class_name = name_split[1]
                    else:
                        class_name = name_split[1] + "/" + name_split[2]
                    self.classes.append(class_name)
        
                img_paths.append(os.path.join(data_dir, row[0]))
        self.all_img_paths = np.asarray(self.all_img_paths)
        self.classes = np.asarray(self.classes)
        
        fix_class_dict = {'bakery/shop': 'bakery',
                          'desert/sand': 'desert_sand',
                          'desert/vegetation': 'vegetation',
                          'dinette/home': 'dinette',
                          'field/cultivated': 'cultivated_field',
                          'field/wild': 'wild',
                          'stadium/baseball': 'baseball_stadium',
                          'stadium/football': 'football_stadium',
                          'temple/east_asia': 'hindu_temple',
                          'temple/south_asia': 'buddhist_temple',
                          'train_station/platform': 'train_platform',
                          'underwater/coral_reef': 'coral_reef'}
        for i in range(len(self.classes)):
            if self.classes[i].endswith('/outdoor'):
                self.classes[i] = self.classes[i][:-8]
            if self.classes[i].endswith('/indoor'):
                self.classes[i] = self.classes[i][:-7]
            if self.classes[i] in fix_class_dict:
                self.classes[i] = fix_class_dict[self.classes[i]]


        self.test_img_paths = []
        self.test_labels = []
        with open(os.path.join(dataset_dir, 'trainvalsplit', 'val_places205.csv'), 'r') as f:
            data = csv.reader(f, delimiter=' ')
            for row in data:
                self.test_img_paths.append(os.path.join(data_dir, row[0]))
                self.test_labels.append(int(row[1]))
        self.test_img_paths = np.asarray(self.test_img_paths)
        self.test_labels = np.asarray(self.test_labels)
        
        self._init_random()
                
        # randomly sample 1000 images since places205 has so/too many images for each class
        for i in range(len(self.all_img_paths)):
            self.all_img_paths[i] = np.random.choice(self.all_img_paths[i], 1000, replace=False)
            
        self.all_img_paths = np.asarray(self.all_img_paths)
        self.classes = np.asarray(self.classes)

        self.train_indices = [np.random.permutation(1000) for _ in range(10)]
