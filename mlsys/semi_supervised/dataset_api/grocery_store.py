import csv
import os
import numpy as np

from .dataset_api import DatasetAPI


class GroceryStore(DatasetAPI):
    def __init__(self, dataset_dir, seed=0):
        super().__init__(dataset_dir, seed)

        fine_grained = self._get_fine_grained()
        
        self.checkpoint_shot = [1, 5]
        
        if fine_grained:
            num_cls = 81
        else:
            num_cls = 43
        
        self.classes = [None] * num_cls
        with open(os.path.join(dataset_dir, 'classes.csv')) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                if fine_grained:
                    self.classes[int(row[1])] = row[0].lower().replace('-', '_')
                else:
                    self.classes[int(row[3])] = row[2].lower().replace('-', '_')
        self.classes = np.asarray(self.classes)

        self.all_img_paths = [[] for i in range(num_cls)]
        with open(os.path.join(dataset_dir, 'train.txt')) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                if fine_grained:
                    self.all_img_paths[int(row[1])].append(row[0])
                else:
                    self.all_img_paths[int(row[2])].append(row[0])
        for i in range(len(self.all_img_paths)):
            self.all_img_paths[i] = np.asarray(self.all_img_paths[i])

        self.test_img_paths = []
        self.test_labels = []
        with open(os.path.join(dataset_dir, 'test.txt')) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                self.test_img_paths.append(row[0])
                if fine_grained:
                    self.test_labels.append(int(row[1]))
                else:
                    self.test_labels.append(int(row[2]))
        self.test_img_paths = np.asarray(self.test_img_paths)
        self.test_labels = np.asarray(self.test_labels)

        self._init_random()
        
        self.train_indices = []
        for i in range(len(self.classes)):
            self.train_indices.append(np.random.permutation(len(self.all_img_paths[i])))

    def _get_fine_grained(self):
        raise NotImplementedError


class GroceryStoreFineGrained(GroceryStore):
    def _get_fine_grained(self):
        return True


class GroceryStoreCoarseGrained(GroceryStore):
    def _get_fine_grained(self):
        return False