import time
import logging
import argparse
import datetime
import pickle
import os
import torch
import numpy as np
import torchvision.models as models
from torch.utils.data import DataLoader
import warnings
from accelerate import Accelerator
accelerator = Accelerator()

from taglets.task import Task
from taglets.controller import Controller
from taglets.task.utils import labels_to_concept_ids
from taglets.scads import Scads
from taglets.labelmodel import AMCLLogReg, AMCLWeightedVote, WeightedVote, UnweightedVote, NaiveBayes

from .dataset_api import FMD, Places205, OfficeHomeProduct, OfficeHomeClipart


log = logging.getLogger(__name__)
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)


class CheckpointRunner:
    def __init__(self, dataset, dataset_dir, batch_size, load_votes_path=None, labelmodel_type=None):
        self.dataset = dataset
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.load_votes_path = load_votes_path
        self.labelmodel_type = labelmodel_type
        
        dataset_dict = {'fmd': FMD,
                        'places205': Places205,
                        'office_home-product': OfficeHomeProduct,
                        'office_home-clipart': OfficeHomeClipart}
        self.dataset_api = dataset_dict[dataset](dataset_dir)

        self.initial_model = models.resnet50(pretrained=True)
        self.initial_model.fc = torch.nn.Identity()
        
        if not os.path.exists('saved_vote_matrices') and accelerator.is_local_main_process:
            os.makedirs('saved_vote_matrices')
        accelerator.wait_for_everyone()
        self.vote_matrix_dict = {}
        self.vote_matrix_save_path = os.path.join('saved_vote_matrices',
                                                  datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    def run_checkpoints(self):
        log.info("Enter checkpoint")
        num_checkpoints = self.dataset_api.get_num_checkpoints()
        for i in range(num_checkpoints):
            self.run_one_checkpoint(i)

    def run_one_checkpoint(self, checkpoint_num):
        log.info('------------------------------------------------------------')
        log.info('--------------------Checkpoint: {}'.format(checkpoint_num)+'---------------------------')
        log.info('------------------------------------------------------------')

        start_time = time.time()
        
        class_names = self.dataset_api.get_class_names()
        
        unlabeled_train_labels = self.dataset_api.get_unlabeled_labels(checkpoint_num)
        labeled_dataset, val_dataset = self.dataset_api.get_labeled_dataset(checkpoint_num)
        unlabeled_train_dataset, unlabeled_test_dataset = self.dataset_api.get_unlabeled_dataset(checkpoint_num)

        evaluation_dataset = self.dataset_api.get_test_dataset()
        test_labels = self.dataset_api.get_test_labels()
        task = Task(self.dataset,
                    ['/c/en/furniture', '/c/en/paper', '/c/en/stationery', '/c/en/appliances',
                     '/c/en/tools', '/c/en/kitchenware',
                     '/c/en/cleaning_implement', '/c/en/sports_equipment',
                     '/c/en/signs', '/c/en/garden'],
                    (224, 224), 
                    labeled_dataset,
                    unlabeled_train_dataset,
                    val_dataset,
                    self.batch_size,
                    None,
                    'predefined/scads.imagenet22k.sqlite3',
                    'predefined/embeddings/numberbatch-en19.08.txt.gz',
                    'predefined/embeddings/imagenet22k_processed_numberbatch.h5',
                    unlabeled_test_data=unlabeled_test_dataset,
                    unlabeled_train_labels=unlabeled_train_labels,
                    test_data=evaluation_dataset,
                    test_labels=test_labels)
        task.set_initial_model(self.initial_model)
        controller = Controller(task)
        
        if self.load_votes_path is None:
            weak_labels = None
        else:
            weak_labels = self._get_weak_labels(checkpoint_num)

        # end_model = controller.train_end_model(weak_labels)
        #
        # if self.vote_matrix_save_path is not None:
        #     val_vote_matrix, unlabeled_vote_matrix = controller.get_vote_matrix()
        #     if val_dataset is not None:
        #         val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
        #         val_labels = [image_labels for _, image_labels in val_loader]
        #     else:
        #         val_labels = None
        #     checkpoint_dict = {'val_images_votes': val_vote_matrix,
        #                        'val_images_labels': val_labels,
        #                        'unlabeled_images_votes': unlabeled_vote_matrix,
        #                        'unlabeled_images_labels': unlabeled_train_labels}
        #     self.vote_matrix_dict[checkpoint_num] = checkpoint_dict
        #     with open(self.vote_matrix_save_path, 'wb') as f:
        #         pickle.dump(self.vote_matrix_dict, f)
        #
        # outputs = end_model.predict(evaluation_dataset)
        # predictions = np.argmax(outputs, 1)
        
        test_vote_matrix = controller.train_end_model(weak_labels)

        categories_dict = {
            '/c/en/furniture': ['Bed', 'Chair', 'Couch', 'Curtains', 'File_Cabinet',
                                'Refrigerator', 'Shelf', 'Sink', 'Table', 'TV', 'Desk_Lamp', 'Lamp_Shade', 'Paper_Clip',
                                'Push_Pin', 'Soda', 'Toys'],
            '/c/en/paper': ['Calendar', 'Clipboards', 'Folder', 'Notebook', 'Post_It_Notes', 'Desk_Lamp', 'Lamp_Shade',
                            'Paper_Clip', 'Push_Pin', 'Soda', 'Toys'],
            '/c/en/stationery': ['Batteries', 'Eraser', 'Marker', 'Pen', 'Pencil', 'Ruler', 'Desk_Lamp', 'Lamp_Shade',
                                 'Paper_Clip', 'Push_Pin', 'Soda', 'Toys'],
            '/c/en/appliances': ['Alarm_Clock', 'Batteries', 'Calculator', 'Computer', 'Fan', 'Keyboard',
                                 'Laptop', 'Monitor', 'Mouse', 'Oven', 'Printer', 'Radio', 'Refrigerator', 'Speaker',
                                 'Telephone', 'TV', 'Webcam', 'Desk_Lamp', 'Lamp_Shade', 'Paper_Clip', 'Push_Pin',
                                 'Soda', 'Toys'],
            '/c/en/tools': ['Drill', 'Hammer', 'Knives', 'Ruler', 'Scissors', 'Screwdriver', 'Desk_Lamp', 'Lamp_Shade',
                            'Paper_Clip', 'Push_Pin', 'Soda', 'Toys'],
            '/c/en/kitchenware': ['Bottle', 'Bucket', 'Candles', 'Fork', 'Kettle', 'Knives', 'Mug', 'Pan', 'Shelf',
                                  'Sink',
                                  'Spoon', 'Trash_Can', 'Desk_Lamp', 'Lamp_Shade', 'Paper_Clip', 'Push_Pin', 'Soda',
                                  'Toys'],
            '/c/en/cleaning_implement': ['Bottle', 'Bucket', 'Mop', 'Sink', 'Toothbrush', 'Trash_Can', 'Desk_Lamp',
                                         'Lamp_Shade', 'Paper_Clip', 'Push_Pin', 'Soda', 'Toys'],
            '/c/en/sports_equipment': ['Backpack', 'Bike', 'Flipflops', 'Glasses', 'Helmet', 'Sneakers', 'Desk_Lamp',
                                       'Lamp_Shade', 'Paper_Clip', 'Push_Pin', 'Soda', 'Toys'],
            '/c/en/signs': ['Batteries', 'Calendar', 'Eraser', 'Exit_Sign', 'Desk_Lamp', 'Lamp_Shade', 'Paper_Clip',
                            'Push_Pin', 'Soda', 'Toys'],
            '/c/en/garden': ['Flowers', 'Desk_Lamp', 'Lamp_Shade', 'Paper_Clip', 'Push_Pin', 'Soda', 'Toys'],
        }
        log.info(f'Categories dict {categories_dict}')
        for idx, concept in enumerate(task.classes):
            categories_dict[idx] = categories_dict.pop(concept)
            categories_dict[idx] = [np.where(class_names == cls.lower())[0][0] for cls in categories_dict[idx]]
            
        binary_categories_dict = categories_dict.copy()
        for k in binary_categories_dict:
            binary_categories_dict[k] = binary_categories_dict[k][:-6]
            
        
        if test_labels is not None:
            log.info('Accuracy of taglets on this checkpoint:')
            # acc = np.sum(predictions == test_labels) / len(test_labels)
            # log.info('Acc {:.4f}'.format(acc))

            for i in range(len(test_vote_matrix)):
                ct = 0
                for j in range(len(test_labels)):
                    pred = np.argmax(test_vote_matrix[i][j])
                    # if i == 0:
                    #     log.info(f'Test {class_names[test_labels[j]]} Pred {task.classes[pred]}')
                    #     if test_labels[j] in categories_dict[pred]:
                    #         ct += 1
                    # else:
                    #     log.info(f'Test {class_names[test_labels[j]]} Pred {task.classes[i - 1]} {pred}')
                    #     if int(test_labels[j] in binary_categories_dict[i - 1]) == pred:
                    #         ct += 1
                    log.info(f'Test {class_names[test_labels[j]]} Pred {task.classes[i]} {pred}')
                    if int(test_labels[j] in binary_categories_dict[i]) == pred:
                        ct += 1
                acc = ct / len(test_labels)
                log.info("Module {} - acc {:.4f}".format(i, acc))

        if self.vote_matrix_save_path is not None:
            val_vote_matrix, unlabeled_vote_matrix = controller.get_vote_matrix()
            if val_dataset is not None:
                val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
                val_labels = [image_labels for _, image_labels in val_loader]
            else:
                val_labels = None
            checkpoint_dict = {'val_images_votes': val_vote_matrix,
                               'val_images_labels': val_labels,
                               'unlabeled_images_votes': unlabeled_vote_matrix,
                               'unlabeled_images_labels': unlabeled_train_labels}
            self.vote_matrix_dict[checkpoint_num] = checkpoint_dict
            with open(self.vote_matrix_save_path, 'wb') as f:
                pickle.dump(self.vote_matrix_dict, f)

        log.info('Checkpoint: {} Elapsed Time =  {}'.format(checkpoint_num,
                                                            time.strftime("%H:%M:%S",
                                                                          time.gmtime(time.time()-start_time))))

    def _load_votes(self, checkpoint_num):
        '''
        Function to get the data from the DARPA task
        '''
    
        data = pickle.load(open(os.path.join("./saved_vote_matrices", self.load_votes_path), "rb"))
    
        data_dict = data[checkpoint_num]
    
        val_votes = data_dict["val_images_votes"]
        val_labels = data_dict["val_images_labels"]
        ul_votes = data_dict["unlabeled_images_votes"]
        ul_labels = data_dict["unlabeled_images_labels"]
    
        if val_votes is not None and val_labels is not None:
            val_votes, val_labels = np.asarray(val_votes), np.asarray(val_labels)
        ul_votes, ul_labels = np.asarray(ul_votes), np.asarray(ul_labels)
    
        # indices = np.arange(len(ul_labels))
        # np.random.shuffle(indices)
        # num_labeled_data = 2000
        # val_votes = ul_votes[:, indices[:num_labeled_data]]
        # val_labels = ul_labels[indices[:num_labeled_data]]
        # ul_votes = ul_votes[:, indices[num_labeled_data:]]
        # ul_labels = ul_labels[indices[num_labeled_data:]]
    
        return val_votes, val_labels, ul_votes, ul_labels
    
    def _get_weak_labels(self, checkpoint_num):
        if accelerator.is_local_main_process:
            val_votes, val_labels, ul_votes, ul_labels = self._load_votes(checkpoint_num)
            if val_votes is not None:
                for i in range(val_votes.shape[0]):
                    log.info(f'Val acc for module {i}: {np.mean(np.argmax(val_votes[i], 1) == val_labels)}')
    
            labelmodel_dict = {'amcl-cc': AMCLWeightedVote,
                               'naive_bayes': NaiveBayes,
                               'weighted': WeightedVote,
                               'unweighted': UnweightedVote}
            num_classes = 10 if self.dataset == 'fmd' else 65
            
            labelmodel = labelmodel_dict[self.labelmodel_type](num_classes)
    
            if self.labelmodel_type == 'amcl-cc':
                if val_votes is None:
                    raise ValueError('Val votes cannot be None')
                labelmodel.train(val_votes, val_labels, ul_votes)
                log.info(f'Thetas: {labelmodel.theta}')
    
            if self.labelmodel_type == 'weighted':
                preds = labelmodel.get_weak_labels(ul_votes,
                                                   [np.mean(np.argmax(val_votes[i], 1) == val_labels) for i in
                                                    range(val_votes.shape[0])])
            else:
                preds = labelmodel.get_weak_labels(ul_votes)

            with open('tmp_labelmodel_output.pkl', 'wb') as f:
                pickle.dump(preds, f)
                
            predictions = np.asarray([np.argmax(pred) for pred in preds])
            log.info("Acc on unlabeled train data %f" % (np.mean(predictions == ul_labels)))

        accelerator.wait_for_everyone()
        with open('tmp_labelmodel_output.pkl', 'rb') as f:
            weak_labels = pickle.load(f)
        accelerator.wait_for_everyone()
        if accelerator.is_local_main_process:
            os.remove('tmp_labelmodel_output.pkl')
        return weak_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        type=str,
                        default='fmd',
                        choices=['fmd', 'places205', 'office_home-product', 'office_home-clipart'])
    parser.add_argument('--dataset_dir',
                        type=str,
                        default='true',
                        help='Option to choose whether to execute or not the entire trining pipeline')
    parser.add_argument('--batch_size',
                        type=int,
                        default='32',
                        help='Universal batch size')
    parser.add_argument('--scads_root_path', 
                        type=str,
                        default='/users/wpiriyak/data/bats/datasets')
    parser.add_argument('--load_votes_path',
                        type=str)
    parser.add_argument('--labelmodel_type',
                        type=str)
    args = parser.parse_args()

    Scads.set_root_path(args.scads_root_path)

    runner = CheckpointRunner(args.dataset, args.dataset_dir, args.batch_size, args.load_votes_path,
                              args.labelmodel_type)
    runner.run_checkpoints()
    

if __name__ == "__main__":
    main()

     