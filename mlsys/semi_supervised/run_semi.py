import time
import logging
import argparse
import datetime
import pickle
import os
import torch
import numpy as np
import torchvision.models as models
import warnings
from accelerate import Accelerator
accelerator = Accelerator()


from taglets.task import Task
from taglets.controller import Controller
from taglets.task.utils import labels_to_concept_ids
from taglets.scads import Scads

from .dataset_api import FMD, Places205, OfficeHome


log = logging.getLogger(__name__)
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)


class CheckpointRunner:
    def __init__(self, dataset, dataset_dir, batch_size):
        self.dataset = dataset
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        
        dataset_dict = {'fmd': FMD, 'places205': Places205, 'office_home': OfficeHome}
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
                    labels_to_concept_ids(class_names),
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

        end_model = controller.train_end_model()
        
        if self.vote_matrix_save_path is not None:
            labeled_vote_matrix, unlabeled_vote_matrix = controller.get_vote_matrix()
            labeled_image_labels = labeled_dataset.labels
            checkpoint_dict = {'labeled_images_votes': labeled_vote_matrix,
                               'labeled_images_labels': labeled_image_labels,
                               'unlabeled_images_votes': unlabeled_vote_matrix,
                               'unlabeled_images_labels': unlabeled_train_labels
                               }
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",
                        type=str,
                        default="fmd")
    parser.add_argument("--dataset_dir",
                        type=str,
                        default="true",
                        help="Option to choose whether to execute or not the entire trining pipeline")
    parser.add_argument("--batch_size",
                        type=int,
                        default="32",
                        help="Universal batch size")
    parser.add_argument('--scads_root_path', 
                        type=str,
                        default='/users/wpiriyak/data/bats/datasets')
    args = parser.parse_args()

    Scads.set_root_path(args.scads_root_path)

    runner = CheckpointRunner(args.dataset, args.dataset_dir, args.batch_size)
    runner.run_checkpoints()
    

if __name__ == "__main__":
    main()

     