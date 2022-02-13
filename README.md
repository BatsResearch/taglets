# TAGLETS:  A System for Automatic Semi-Supervised Learning with Auxiliary Data

## Installation

The package requires `python3.7`. You first need to clone this repository:
```
git clone https://github.com/BatsResearch/taglets.git
```

Before installing TAGLETS, we recommend creating and activating a new virtual environment which can be done by the following script:
```
python -m venv taglets_venv
source taglets_venv/bin/activate
```

You also want to make sure `pip` is up-to-date:
```
pip install --upgrade pip
```

Then, to install TAGLETS and download related files, run:
```
bash setup.sh
```


## Solve a task (with limited labeled data) with TAGLETS

For this demo, we will assume that our auxiliary dataset is CIFAR-100, and we want to solve CIFAR-10 with only 
0.1% of its training data labeled.

First, download CIFAR-100 and the SCADS file with CIFAR-100 installed by running the below bash script at the outermost 
directory of TAGLETS 

```
mkdir aux_data
cd aux_data
wget -nc https://storage.googleapis.com/taglets-public/cifar100.zip
unzip cifar100.zip
rm cifar100.zip
cd ../predefined
wget -nc https://storage.googleapis.com/taglets-public/scads.cifar100.sqlite3
cd embeddings
wget -nc https://storage.googleapis.com/taglets-public/embeddings/cifar100_processed_numberbatch.h5
```

Then, run the below python script: 

(We recommend using GPUs to run the script below. Please see the GPU/Multi-GPU support for instructions on how to launch the python script so that it uses GPUs and also 
uncomment the commented part of the code below)

```python
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
```

## GPU/Multi-GPU Support

TAGLETS uses the package `accelerate` to support the use of one or more GPUs. You need to use the `accelerate launcher` to run the script in order to use GPUs. 

Suppose you want to use 4 GPUs. Your config file, e.g., `acclerate_config.yml`, should look similar to this:
```yml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
fp16: false
machine_rank: 0
main_process_ip: null
main_process_port: null
main_training_function: main
num_machines: 1
num_processes: 4 # set this number to the number of gpus
```

Then, you can run the launcher as following:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file accelerate_config.yml run_demo.py
```
where `run_demo.py` contains your python script using TAGLETS

One important thing to note of `accelerate` is it spawns multiple processes running the same python script, so as with other multiprocessing code, you need to keep in mind the usual parallelization caveats. These include but not limited to:

- Make sure the script is deterministic across processes
- When saving or loading files, make sure all processes are joined before doing so
- When interacting with an external server, might want to only do that only in the main process to avoid duplicate requests

We recommend reading more about `accelerate` before you try to use multiple gpus: https://huggingface.co/docs/accelerate/ 
