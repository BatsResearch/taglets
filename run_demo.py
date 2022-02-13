import numpy as np
import torch.nn as nn
from torch.utils.data import Subset, Dataset
from torchvision.datasets import CIFAR10
import torchvision.models as models
import torchvision.transforms as transforms

from taglets import Controller
from taglets.scads import Scads
from taglets.task import Task
from taglets.task.utils import labels_to_concept_ids

# from taglets.models import bit_backbone

# IMPORTANT!!: Uncomment this part of the code if you want to use GPUs
# import random
# from accelerate import Accelerator
# accelerator = Accelerator()
# # We want to avoid non-deterministic behavoirs in our multi-GPU code
# random.seed(0)
# np.random.seed(0)
# # If multiple processes try to download CIFAR10 to the filesytem at once, you might get an error
# # So we modify the code to download the dataset only in the main process
# if accelerator.is_local_main_process:
#     _ = CIFAR10('.', train=True, download=True)
#     _ = CIFAR10('.', train=False, download=True)
# accelerator.wait_for_everyone()

# ---------------- Setting up an example task with limited labeled data ---------------
# This example task is CIFAR10, but only 0.1% of the training data is labeled.
# The rest of the training data is used as unlabeled examples.

data_mean = [0.485, 0.456, 0.406]
data_std = [0.229, 0.224, 0.225]
train_transform = transforms.Compose([
    transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=data_mean, std=data_std)
])
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=data_mean, std=data_std)
])
train_dataset = CIFAR10('.', train=True, transform=train_transform, download=True)
test_dataset = CIFAR10('.', train=True, transform=test_transform, download=True)

labeled_percent = 0.001
num_train_data = 50000
indices = list(range(num_train_data))
train_split = int(np.floor(labeled_percent * num_train_data))
np.random.shuffle(indices)
labeled_idx = indices[:train_split]
unlabeled_idx = indices[train_split:]
labeled_dataset = Subset(train_dataset, labeled_idx)
unlabeled_dataset = Subset(train_dataset, unlabeled_idx)

# Make sure TAGLETS will not see the labels of unlabeled data
class HiddenLabelDataset(Dataset):
    """
    Wraps a labeled dataset so that it appears unlabeled
    """
    def __init__(self, dataset):
        self.subset = dataset
        self.dataset = self.subset.dataset

    def __getitem__(self, idx):
        data = self.subset[idx]
        try:
            img1, img2, _ = data
            return img1, img2

        except ValueError:
            return data[0]

    def __len__(self):
        return len(self.subset)
unlabeled_dataset = HiddenLabelDataset(unlabeled_dataset)

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog',
               'horse', 'ship', 'truck']
# You can either use our utility function to automatically map class names to concepts,
# Or you can do it manually
concepts = labels_to_concept_ids(class_names)

# --------------------------------------------------------------------------------------

# Set the path where your auxiliary datasets are at
Scads.set_root_path('aux_data')

# Choose your backbone - we support ResNet50 and Bit-ResNet50v2
initial_model = models.resnet50(pretrained=True)
initial_model.fc = nn.Identity()
# We provide BigTransfer using resnet50v2 pre-trained on ImageNet-21k:
# initial_model = bit_backbone()

# Configure your Task instance
# SCADS and SCADS Embeddings files for the setup of SCADS used in the paper (ConceptNet + ImageNet21k)
# is automatically downloaded when you install and set up TAGLETS
scads_path = 'predefined/scads.cifar100.sqlite3' # Path to SCADS file
scads_embedding_path = 'predefined/embeddings/numberbatch-en19.08.txt.gz' # Path to SCADS Embedding file
# Optional (for faster computation): path to processed SCADS Embedding file where all embeddings of nodes without images are removed
processed_scads_embedding_path='predefined/embeddings/cifar100_processed_numberbatch.h5'

task = Task('limited-labeled-cifar10', # Task name
            concepts, # Target concepts
            (224, 224), # Image size
            labeled_dataset, # Training labeled data
            unlabeled_dataset, # Training unlabeled data
            None, # Validation dataset
            32, # Batch size
            scads_path=scads_path, # Path to the SCADS file
            scads_embedding_path=scads_embedding_path, # Path to the SCADS Embeddings file
            processed_scads_embedding_path=processed_scads_embedding_path, # (Optional) Path to
            # the processed SCADS Embeddings file where the nodes without any auxiliary images are pruned
            wanted_num_related_class=3) # Num of auxiliary classes per target class
task.set_initial_model(initial_model)
task.set_model_type('resnet50') # or 'bigtransfer'

# Use the Task instance to create a Controller
# Then, use the Controller to get a trained end model, ready to do prediction
controller = Controller(task)
end_model = controller.train_end_model()

# Use the trained end model to get predictions
outputs, _ = end_model.predict(test_dataset)
predictions = np.argmax(outputs, 1)

# Or get the end model's accuracy on the test data
print(f'Accuracy on the test data = {end_model.evaluate(test_dataset)}')