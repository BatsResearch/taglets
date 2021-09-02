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

from taglets.scads import Scads
from taglets.labelmodel import AMCLLogReg, AMCLWeightedVote, WeightedVote, UnweightedVote, NaiveBayes

from .dataset_api import FMD, Places205, OfficeHomeProduct, OfficeHomeClipart
from .models import KNOWN_MODELS

log = logging.getLogger(__name__)
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1)


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

        model = KNOWN_MODELS['BiT-M-R50x1']()
        model.load_from(np.load("BiT-M-R50x1.npz"))
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

        evaluation_dataset = self.dataset_api.get_test_dataset()
        test_labels = self.dataset_api.get_test_labels()
        
        mapping = {'alarm_clock': ['4234'], 'backpack': ['4479'], 'batteries': ['4664'], 'bed': ['4706', '4707'],
         'bike': ['4774'], 'bottle': ['4961'], 'bucket': ['5112'], 'calculator': ['5235'], 'calendar': ['6179'],
         'candles': ['5276'], 'chair': ['5515', '5516'], 'clipboards': ['21842'], 'computer': ['5848'],
         'couch': ['5977', '5978'], 'curtains': ['6129'], 'desk_lamp': ['8087', '8088'], 'drill': ['6478'],
         'eraser': ['6696'], 'exit_sign': ['12241'], 'fan': ['6774'], 'file_cabinet': ['6851'], 'flipflops': ['6955'],
         'flowers': ['17522'], 'folder': ['7000'], 'fork': ['7038'], 'glasses': ['10621'], 'hammer': ['7446', '7448'],
         'helmet': ['7592', '7593'], 'kettle': ['7989'], 'keyboard': ['7993'], 'knives': ['8039', '8040'],
         'lamp_shade': ['8091'], 'laptop': ['8113'], 'marker': ['8422'], 'monitor': ['8627', '8628'], 'mop': ['11004'],
         'mouse': ['8675'], 'mug': ['8693'], 'notebook': ['8816'], 'oven': ['8937'], 'pan': ['9007', '9008'],
         'paper_clip': ['9035'], 'pen': ['9124'], 'pencil': ['9129', '9130'], 'printer': ['9502', '9503'],
         'push_pin': ['11267'], 'radio': ['9644'], 'refrigerator': ['6622'], 'ruler': ['9964'], 'scissors': ['10103'],
         'screwdriver': ['10125'], 'shelf': ['10266'], 'sink': ['14403'], 'sneakers': ['7408'],
         'soda': ['14039', '14040'], 'speaker': ['8290'], 'spoon': ['10673', '10674'], 'table': ['11052', '11053'],
         'telephone': ['11152'], 'toothbrush': ['11356'], 'toys': ['11391'], 'trash_can': ['4379'], 'tv': ['11169'],
         'webcam': ['11792'], 'post_it_notes': ['12165']}

        class_indices = []
        inverse_mapping = {}
        for cls, ss in mapping.items():
            for s in ss:
                inverse_mapping[len(class_indices)] = np.where(class_names == cls)[0][0]
                class_indices.append(int(s) - 1)
                
        def get_preds(dataset):
            self.initial_model.cuda()
            dataloader = DataLoader(dataset, shuffle=False, batch_size=256)
            outputs = []
            for batch in dataloader:
                batch = batch.cuda()
                with torch.no_grad():
                    logits = self.initial_model(batch)
                    outputs.append(logits[:, class_indices].detach().cpu())
            outputs = torch.cat(outputs).numpy()
            outputs = np.reshape(outputs, (len(outputs), -1))
            predictions = np.zeros((len(outputs), len(class_names)))
            for i in range(len(predictions)):
                for j in range(outputs.shape[1]):
                    predictions[i][inverse_mapping[j]] = max(predictions[i][inverse_mapping[j]], outputs[i][j])
                predictions[i] = softmax(predictions[i])
            return predictions
        
        if test_labels is not None:
            log.info('Accuracy of taglets on this checkpoint:')
            preds = get_preds(evaluation_dataset)
            predictions = np.argmax(preds, axis=-1)
            acc = np.sum(predictions == test_labels) / len(test_labels)
            log.info('Acc {:.4f}'.format(acc))

        if self.vote_matrix_save_path is not None:
            unlabeled_train_labels = self.dataset_api.get_unlabeled_labels(checkpoint_num)
            _, unlabeled_test_dataset = self.dataset_api.get_unlabeled_dataset(checkpoint_num)
            preds = get_preds(unlabeled_test_dataset)
            checkpoint_dict = {'unlabeled_images_votes': np.asarray([preds]),
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

    