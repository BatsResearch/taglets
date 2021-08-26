import random
import copy
import os
import torch
import logging
import torchvision.transforms as transforms
import torch.nn as nn
from nltk.corpus import wordnet as wn
from accelerate import Accelerator

accelerator = Accelerator()

from .module import Module
from ..pipeline import ImageTagletWithAuxData, Cache
from ..data.custom_dataset import CustomImageDataset
from ..scads import Scads, ScadsEmbedding

log = logging.getLogger(__name__)


class BinaryPartialModule(Module):
    """
    A module that pre-trains on datasets selected from the SCADS and then
    transfers to available labeled data
    """
    
    def __init__(self, task):
        super().__init__(task)
        self.taglets = [BinaryPartialTaglet(task, i) for i in range(len(task.classes))]


class BinaryPartialTaglet(ImageTagletWithAuxData):
    def __init__(self, task, class_idx):
        super().__init__(task)
        self.name = 'binary-partial'
        if os.getenv("LWLL_TA1_PROB_TASK") is not None:
            self.save_dir = os.path.join('/home/tagletuser/trained_models', self.name)
        else:
            self.save_dir = os.path.join('trained_models', self.name)
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.class_idx = class_idx
        
        self.num_related_class = 1
        self.prune = -1
    
    def _set_num_classes(self, num_classes):
        m = torch.nn.Sequential(*list(self.model.children())[:-1])
        output_shape = self._get_model_output_shape(self.task.input_shape, m)
        self.model.fc = torch.nn.Linear(output_shape, num_classes)
        
        params_to_update = []
        for param in self.model.parameters():
            if param.requires_grad:
                params_to_update.append(param)
        self._params_to_update = params_to_update
        self.optimizer = torch.optim.SGD(self._params_to_update, lr=0.003, momentum=0.9)
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[3, 4], gamma=0.1)
    
    def transform_image(self, train=True):
        """
        Get the transform to be used on an image.
        :return: A transform
        """
        data_mean = [0.485, 0.456, 0.406]
        data_std = [0.229, 0.224, 0.225]
        
        if train:
            return transforms.Compose([
                transforms.RandomResizedCrop(self.task.input_shape, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=data_mean, std=data_std)
            ])
        else:
            return transforms.Compose([
                transforms.Resize(self.task.input_shape),
                transforms.ToTensor(),
                transforms.Normalize(mean=data_mean, std=data_std)
            ])

    def _get_scads_data(self):
        data = Cache.get(f"scads-binary-partial-{self.class_idx}", self.task.classes)
        if data is not None:
            image_paths, image_labels, all_related_class = data
        else:
            root_path = Scads.get_root_path()
            Scads.open(self.task.scads_path)
            ScadsEmbedding.load(self.task.scads_embedding_path, self.task.processed_scads_embedding_path)
            image_paths = []
            image_labels = []
            visited = set()
        
            target_synsets = []
            for conceptnet_id in self.task.classes:
                class_name = conceptnet_id[6:]
                target_synsets = target_synsets + wn.synsets(class_name, pos='n')
        
            def get_images(node, label, is_neighbor):
                if is_neighbor and node.get_conceptnet_id() in self.task.classes:
                    return False
                if node.get_conceptnet_id() not in visited:
                    visited.add(node.get_conceptnet_id())
                
                    synsets = wn.synsets(node.get_conceptnet_id()[6:], pos='n')
                    for synset in synsets:
                        for target_synset in target_synsets:
                            lch = synset.lowest_common_hypernyms(target_synset)[0]
                            if target_synset.shortest_path_distance(lch) <= self.prune:
                                return False
                
                    images = node.get_images_whitelist(self.task.whitelist)
                    if len(images) < self.img_per_related_class:
                        return False
                
                    bad_images = ['imagenet/n06470073/n06470073_47249.JPEG', 'imagenet/n04135315/n04135315_8814.JPEG',
                                  'imagenet/n04257684/n04257684_9033.JPEG']
                
                    images = random.sample(images, self.img_per_related_class)
                    images = [os.path.join(root_path, image) for image in images if image not in bad_images]
                    if label:
                        for i in range(len(self.task.classes) - 1):
                            image_paths.extend(images)
                            image_labels.extend([label] * len(images))
                    else:
                        image_paths.extend(images)
                        image_labels.extend([label] * len(images))
                    log.info("Source class found: {}".format(node.get_conceptnet_id()))
                    return True
                return False
        
            for idx, conceptnet_id in enumerate(self.task.classes):
                log.info(f'Finding source class for {conceptnet_id}')
                cur_related_class = 0
                target_node = Scads.get_node_by_conceptnet_id(conceptnet_id)
                if get_images(target_node, int(idx == self.class_idx), False):
                    cur_related_class += 1
            
                ct = 1
                while cur_related_class < self.num_related_class:
                    processed_embeddings_exist = (self.task.processed_scads_embedding_path is not None)
                    neighbors = ScadsEmbedding.get_related_nodes(target_node,
                                                                 limit=self.num_related_class * 10 * ct,
                                                                 only_with_images=processed_embeddings_exist)
                    for neighbor in neighbors:
                        if get_images(neighbor, int(idx == self.class_idx), True):
                            cur_related_class += 1
                            if cur_related_class >= self.num_related_class:
                                break
                    ct = ct * 2

            all_related_class = 2
        
            Scads.close()
            Cache.set(f"scads-binary-partial-{self.class_idx}", self.task.classes,
                      (image_paths, image_labels, all_related_class))
    
        transform = self.transform_image(train=True)
        train_dataset = CustomImageDataset(image_paths,
                                           labels=image_labels,
                                           transform=transform)
    
        return train_dataset, all_related_class
    
    def train(self, train_data, val_data, unlabeled_data=None):
        scads_train_data, scads_num_classes = self._get_scads_data()
        log.info("Source classes found: {}".format(scads_num_classes))
        self._set_num_classes(scads_num_classes)
        aux_weights = Cache.get(f"scads-weights-binary-partial-{self.class_idx}", self.task.classes)
        if aux_weights is None:
            orig_num_epochs = self.num_epochs
            self.num_epochs = 5 if not os.environ.get("CI") else 5
            super().train(scads_train_data, None, None)
            self.num_epochs = orig_num_epochs
            
            # self.model.fc = nn.Identity()
            aux_weights = copy.deepcopy(self.model.state_dict())
            Cache.set(f'scads-weights-binary-partial-{self.class_idx}', self.task.classes, aux_weights)
        self.model.load_state_dict(aux_weights, strict=False)

