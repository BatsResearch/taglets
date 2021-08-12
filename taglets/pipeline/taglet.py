import os
import random
import logging
import numpy as np

from .trainable import ImageTrainable, VideoTrainable


log = logging.getLogger(__name__)

class TagletMixin:
    def execute(self, unlabeled_data):
        """
        Execute the Taglet on unlabeled images.

        :param unlabeled_data: A Dataset containing unlabeled data
        :return: A 1-d NumPy array of predicted labels
        """
        
        outputs = self.predict(unlabeled_data)
        if self.name == 'svc-video':
            return outputs
        
        return np.argmax(outputs, 1)

class AuxDataMixin:
    def __init__(self, task):
        super().__init__(task)
        self.img_per_related_class = 600 if not os.environ.get("CI") else 1
        if os.environ.get("CI"):
            self.num_related_class = 1
        else:
            self.num_related_class = 10 if len(self.task.classes) < 100 else (5 if len(self.task.classes) < 300 else 3)
    
    def _get_scads_data(self):
        data = Cache.get("scads", self.task.classes)
        if data is not None:
            image_paths, image_labels, all_related_class = data
        else:
            root_path = Scads.get_root_path()
            Scads.open(self.task.scads_path)
            ScadsEmbedding.load(self.task.scads_embedding_path, self.task.processed_scads_embedding_path)
            image_paths = []
            image_labels = []
            visited = set()
        
            def get_images(node, label, is_neighbor):
                if is_neighbor and node.get_conceptnet_id() in self.task.classes:
                    return False
                if node.get_conceptnet_id() not in visited:
                    visited.add(node.get_conceptnet_id())
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
                if get_images(target_node, all_related_class, False):
                    cur_related_class += 1
                    all_related_class += 1

                ct = 1
                while cur_related_class < self.num_related_class:
                    processed_embeddings_exist = (self.task.processed_scads_embedding_path is not None)
                    neighbors = ScadsEmbedding.get_related_nodes(target_node,
                                                                 limit=self.num_related_class * 10 * ct,
                                                                 only_with_images=processed_embeddings_exist)
                    for neighbor in neighbors:
                        if get_images(neighbor, all_related_class, True):
                            cur_related_class += 1
                            all_related_class += 1
                            if cur_related_class >= self.num_related_class:
                                break
                    ct = ct * 2
        
            Scads.close()
            Cache.set('scads', self.task.classes,
                      (image_paths, image_labels, all_related_class))
    
        transform = self.transform_image(train=True)
        train_dataset = CustomImageDataset(image_paths,
                                           labels=image_labels,
                                           transform=transform)
    
        return train_dataset, all_related_class

class VideoAuxDataMixin(AuxDataMixin):
    def __init__(self, task):
        super().__init__(task)
        
        self.img_per_related_class = 20 if not os.environ.get("CI") else 1
        if os.environ.get("CI"):
            self.num_related_class = 1
        else:
            self.num_related_class = 10 if len(self.task.classes) < 30 else (5 if len(self.task.classes) < 60 else 3)

    def _get_scads_data(self):
        data = Cache.get("scads", self.task.classes)
        if data is not None:
            image_paths, image_labels, all_related_class = data
        else:
            root_path = Scads.get_root_path()
            Scads.open(self.task.scads_path)
            ScadsEmbedding.load(self.task.scads_embedding_path, self.task.processed_scads_embedding_path)
            clip_paths = []
            clip_labels = []
            dictionary_clips = {}
            visited = set()
        
            def get_clips(node, label, is_neighbor):
                if is_neighbor and node.get_conceptnet_id() in self.task.classes:
                    return False
                if node.get_conceptnet_id() not in visited:
                    visited.add(node.get_conceptnet_id())
                    clips = node.get_clips_whitelist(self.task.whitelist)
                    if len(clips) < self.img_per_related_class:
                        return False
                    clips = random.sample(clips, self.img_per_related_class)
                    log.info(f"Concept: {clips[0][-1]}")
                    
                    paths = []
                    for path, start, end, v_idx, name_concept in clips:
                        if name_concepts == node.get_conceptnet_id():
                            base_path_clip = os.path.join(root_path, path, v_idx)
                            paths.append(base_path_clip)
                            
                            action_frames = [base_path_clip + '/' + str(i)+'.jpg' for i in range(int(start), int(end) + 1)]
                            dictionary_clips[v_idx] = action_frames

                    log.info(f"Number of clips: {len(paths)} and example: {paths[0]}")
                    
                    clip_paths.extend(paths)
                    clip_labels.extend([label] * len(clips))
                    log.debug("Source class found: {}".format(node.get_conceptnet_id()))
                    return True
                return False
        
            all_related_class = 0
            for conceptnet_id in self.task.classes:
                cur_related_class = 0
                
                try:
                    target_nodes = Scads.get_node_by_conceptnet_id(conceptnet_id)
                except:
                    target = conceptnet_id.split('/')[-1]
                    nodes = [f"/c/en/{w.strip()}" for w in target.split('_')]
                    target_nodes = [Scads.get_node_by_conceptnet_id(n) for n in nodes]
                
                if isinstance(target_nodes, list):
                    target_node = target_nodes[0]
                else:
                    target_node = target_nodes

                if get_clips(target_node, all_related_class, False):
                    cur_related_class += 1
                    all_related_class += 1

                ct = 1
                while cur_related_class < self.num_related_class:
                    processed_embeddings_exist = False#(self.task.processed_scads_embedding_path is not None)

                    target_node = target_nodes                    
                    neighbors = ScadsEmbedding.get_related_nodes(target_node,
                                                                 limit=self.num_related_class * 10 * ct,
                                                                 only_with_images=processed_embeddings_exist)
                    for neighbor in neighbors:
                        if get_clips(neighbor, all_related_class, True):
                            cur_related_class += 1
                            all_related_class += 1
                            if cur_related_class >= self.num_related_class:
                                break
                    ct = ct * 2
        
            Scads.close()
            Cache.set('scads', self.task.classes,
                      (image_paths, image_labels, all_related_class))

        train_dataset = CustomVideoDataset(clip_paths,
                                            labels=clip_labels,
                                            transform_img=self.transform_image(video=self.video),
                                            transform_vid=self.transformer_video(),
                                            clips_dictionary=dictionary_clips)
    
        return train_dataset, all_related_class

class ImageTaglet(ImageTrainable, TagletMixin):
    """
    A trainable model that produces votes for unlabeled images
    """

class ImageTagletWithAuxData(AuxDataMixin, TagletMixin, ImageTrainable):
    """
    A trainable model that produces votes for unlabeled images and uses ScadsEmbedding to get auxiliary data
    """

class VideoTaglet(VideoTrainable, TagletMixin):
    """
    A trainable model that produces votes for unlabeled videos
    """


class VideoTagletWithAuxData(VideoAuxDataMixin, TagletMixin, VideoTrainable):
    """
    A trainable model that produces votes for unlabeled images and uses ScadsEmbedding to get auxiliary data
    """

