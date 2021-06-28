import os
import logging
import numpy as np
import random
from nltk.corpus import wordnet as wn

from .trainable import ImageTrainable, VideoTrainable
from ..scads import Scads, ScadsEmbedding
from ..pipeline import Cache
from ..data.custom_dataset import CustomImageDataset

log = logging.getLogger(__name__)

class TagletMixin:
    def execute(self, unlabeled_data):
        """
        Execute the Taglet on unlabeled images.

        :param unlabeled_data: A Dataset containing unlabeled data
        :return: A 1-d NumPy array of predicted labels
        """
        outputs = self.predict(unlabeled_data)
        return np.argmax(outputs, 1)
    

class ScadsTagletMixin:
    def __init__(self, task):
        super().__init__(task)
        self.img_per_related_class = 600 if not os.environ.get("CI") else 1
        self.num_related_class = 10
        self.prune = 0

    def _get_scads_data(self):
        data = Cache.get("scads", self.task.classes)
        if data is not None:
            image_paths, image_labels, all_related_class = data
        else:
            root_path = Scads.get_root_path()
            Scads.open(self.task.scads_path)
            ScadsEmbedding.load(self.task.scads_embedding_path)
            image_paths = []
            image_labels = []
            visited = set()
        
            target_synsets = []
            for conceptnet_id in self.task.classes:
                class_name = conceptnet_id[6:]
                target_synsets = target_synsets + wn.synsets(class_name, pos='n')
        
            def get_images(node, label):
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
                
                    images = random.sample(images, self.img_per_related_class)
                    images = [os.path.join(root_path, image) for image in images]
                    image_paths.extend(images)
                    image_labels.extend([label] * len(images))
                    log.debug("Source class found: {}".format(node.get_conceptnet_id()))
                    return True
                return False
        
            all_related_class = 0
            for conceptnet_id in self.task.classes:
                cur_related_class = 0
                target_node = Scads.get_node_by_conceptnet_id(conceptnet_id)
                if get_images(target_node, all_related_class):
                    cur_related_class += 1
                    all_related_class += 1
            
                ct = 1
                while cur_related_class < self.num_related_class:
                    neighbors = ScadsEmbedding.get_related_nodes(target_node, self.num_related_class * 20 * ct)
                    for neighbor in neighbors:
                        if get_images(neighbor, all_related_class):
                            cur_related_class += 1
                            all_related_class += 1
                            if cur_related_class >= self.num_related_class:
                                break
                    ct += 1
        
            Scads.close()
            Cache.set('scads', self.task.classes,
                      (image_paths, image_labels, all_related_class))
    
        transform = self.transform_image(train=False)
        train_data = CustomImageDataset(image_paths,
                                        labels=image_labels,
                                        transform=transform)
    
        return train_data, all_related_class


class ImageTaglet(TagletMixin, ImageTrainable):
    """
    A trainable model that produces votes for unlabeled images
    """


class ScadsImageTaglet(ScadsTagletMixin, TagletMixin, ImageTrainable):
    """
    A trainable model that produces votes for unlabeled images and uses ScadsEmbedding to get auxiliary data
    """


class VideoTaglet(VideoTrainable, TagletMixin):
    """
    A trainable model that produces votes for unlabeled videos
    """

