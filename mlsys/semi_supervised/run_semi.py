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

from .dataset_api import FMD, Places205, OfficeHomeProduct, OfficeHomeClipart, GroceryStoreFineGrained, \
    GroceryStoreCoarseGrained
from .models import KNOWN_MODELS


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
                        'office_home-clipart': OfficeHomeClipart,
                        'grocery-coarse': GroceryStoreCoarseGrained,
                        'grocery-fine': GroceryStoreFineGrained}
        self.dataset_api = dataset_dict[dataset](dataset_dir)

        model = KNOWN_MODELS['BiT-M-R50x1'](head_size=len(self.dataset_api.get_class_names()),
                                            zero_head=True)
        model.load_from(np.load("BiT-M-R50x1.npz"))
        model.head.conv = torch.nn.Identity()
        self.initial_model = model
        
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
                    labels_to_concept_ids(class_names),
                    (224, 224), 
                    labeled_dataset,
                    unlabeled_train_dataset,
                    val_dataset,
                    self.batch_size,
                    None,
                    'predefined/scads.imagenet22k-mod.sqlite3',
                    'predefined/embeddings/numberbatch-en19.08.txt.gz',
                    'predefined/embeddings/imagenet22k-mod_processed_numberbatch.h5',
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

        end_model = controller.train_end_model(weak_labels)
        
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
        
        outputs = end_model.predict(evaluation_dataset)
        predictions = np.argmax(outputs, 1)
        
        if test_labels is not None:
            log.info('Accuracy of taglets on this checkpoint:')
            acc = np.sum(predictions == test_labels) / len(test_labels)
            log.info('Acc {:.4f}'.format(acc))

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
            num_classes = 10 if self.dataset == 'fmd' else 65 if self.dataset.startswith('officehome') else 43
            
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
                        choices=['fmd', 'places205', 'office_home-product', 'office_home-clipart',
                                 'grocery-coarse', 'grocery-fine'])
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

     