import os
import sys
import random
import logging
import numpy as np

from .trainable import ImageTrainable, VideoTrainable
from ..scads import Scads, ScadsEmbedding, ScadsEmbeddingVideo
from ..pipeline import Cache
from ..data.custom_dataset import CustomImageDataset

from .trainable import ImageTrainable, VideoTrainable
from ..scads import Scads, ScadsEmbedding
from ..pipeline import Cache
from ..data.custom_dataset import CustomVideoDataset, HandleExceptionCustomVideoDataset

log = logging.getLogger(__name__)

class TagletMixin:
    def execute(self, unlabeled_data, evaluation=False, test=False):
        """
        Execute the Taglet on unlabeled images.

        :param unlabeled_data: A Dataset containing unlabeled data
        :return: A 1-d NumPy array of predicted labels
        """
        
        
        if self.task.video_classification:
            log.info('Executing video-classification')
            if self.name == 'svc-video':
                outputs = self.predict(unlabeled_data, evaluation, test)
                return outputs
            else:
                outputs = self.predict(unlabeled_data)
                return np.argmax(outputs, 1)
        else:
            outputs = self.predict(unlabeled_data)
            if isinstance(outputs, tuple):
                outputs, _ = outputs
            return outputs

class AuxDataMixin:
    def __init__(self, task):
        super().__init__(task)
        self.img_per_related_class = 600 if not os.environ.get("CI") else 1
        if os.environ.get("CI"):
            self.num_related_class = 1
        else:
            self.num_related_class = 5 if len(self.task.classes) < 100 else (3 if len(self.task.classes) < 300 else 1)
    
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
        
        self.img_per_related_class = 100 if not os.environ.get("CI") else 1
        if os.environ.get("CI"):
            self.num_related_class = 1
        else:
            self.num_related_class = 5 if len(self.task.classes) < 30 else (1 if len(self.task.classes) < 60 else 1)

        self.seed = 0
        self._init_random(self.seed)

    @staticmethod
    def _init_random(seed):
        """
        Initialize random numbers with a seed.
        :param seed: The seed to initialize with
        :return: None
        """
        random.seed(seed)
        np.random.seed(seed)


    def _get_scads_data(self, split=False):
        data = Cache.get("scads", self.task.classes)
        if data is not None:
            clip_paths, clip_labels, dictionary_clips, all_related_class = data
        else:
            clip_paths = []
            clip_labels = []
            dictionary_clips = {}
            visited = set()
        
            def get_clips(node, label, is_neighbor):
                extra_ds = ['moments-in-time', 'kinetics700']

                if is_neighbor and node.get_conceptnet_id() in self.task.classes:
                    return False
                if node.get_conceptnet_id() not in visited:
                    visited.add(node.get_conceptnet_id())
                    proc_whitelist = [d.lower() for d in self.task.whitelist] 
                    proc_whitelist.remove('kinetics400')
                    # Get clips for the class
                    clips = node.get_clips_whitelist(proc_whitelist)
                    tmp_clips = [(path, start, end, c_idx, v_idx, name_concept) \
                        for path, start, end, c_idx, v_idx, name_concept in clips \
                        if ('/c/en/' + name_concept == node.get_conceptnet_id()) and (path.split('/')[0] in proc_whitelist)]
                    
                    clips = []
                    for path, start, end, c_idx, v_idx, name_concept in tmp_clips:
                        if (path.split('/')[0] == 'moments-in-time') and (path.split('/')[2] == 'train'):
                            continue
                        elif (path.split('/')[0] == 'kinetics700') and (path.split('/')[2] == 'train'):
                            continue
                        else:
                            clips.append((path, start, end, c_idx, v_idx, name_concept))

                    if len(clips) < 10:#self.img_per_related_class:
                        return False
                    clips = random.sample(clips, min(self.img_per_related_class, len(clips)))

                    paths = []
                    for path, start, end, c_idx, v_idx, name_concept in clips:
                        if '/c/en/' + name_concept == node.get_conceptnet_id():
                            #print(f"root path scads {root_path.split('/')[-1]}, path {path}")
                            split_path = path.split('/')
                            if split_path[0] in extra_ds:
                                # Comment it out for submission
                                base = '/'.join(root_path.split('/')[0:-1]) + '/extra/' 
                                #base = '/'.join(root_path.split('/')[0:-2]) + '/lwll-taglets/extra/' 
                                paths.append(os.path.join(base, path))
                                base_path_clip = os.path.join(base, path)
                                action_frames = [base_path_clip + '/img_' + str(i).zfill(5) +'.jpg' for i in range(int(start), int(end) + 1)]
                                dictionary_clips[str(c_idx)] = action_frames
                            else:
                                paths.append(os.path.join(root_path, path))
                                
                                base_path_clip = os.path.join(root_path, path)
                                action_frames = [base_path_clip + '/' + str(i) +'.jpg' for i in range(int(start), int(end) + 1)]
                                dictionary_clips[str(c_idx)] = action_frames

                    log.info(f"Concept: {node.get_conceptnet_id()} and {'/c/en/' + name_concept} and length paths: {len(paths)}")
                    
                    clip_paths.extend(paths)
                    clip_labels.extend([label] * len(clips))
                    return True
                return False
            
            # Modify for multiple roots
            root_path = Scads.get_root_path()
            Scads.open(self.task.scads_path)
            ScadsEmbeddingVideo.load(self.task.scads_embedding_path, self.task.processed_scads_embedding_path)
            all_related_class = 0
            
            for conceptnet_id in self.task.classes:
                #print(conceptnet_id)
                cur_related_class = 0
                
                try:
                    target_nodes = Scads.get_node_by_conceptnet_id(conceptnet_id)
                except:
                    #log.info(f'Class node immediately found: {target_nodes.node}')
                    target = conceptnet_id.split('/')[-1]
                    nodes = [f"/c/en/{w.strip()}" for w in target.split('_')]
                    target_nodes = []
                    for n in nodes:
                        try:
                            target_nodes.append(Scads.get_node_by_conceptnet_id(n))
                        except:
                            continue
                    #log.info(f'Class nodes from compound: {target_nodes}')
                
                if isinstance(target_nodes, list) == False:
                    #log.info(f"Unique target node: {conceptnet_id}")
                    target_node = target_nodes
                    if get_clips(target_node, all_related_class, False):
                        cur_related_class += 1
                        all_related_class += 1

                ct = 1
                iters = 0
                while cur_related_class < self.num_related_class and iters <= 10:
                    processed_embeddings_exist = (self.task.processed_scads_embedding_path is not None)
                    #print(f"path processed scads {processed_embeddings_exist}")

                    #print(f"{conceptnet_id} and nodes {target_nodes}")
                    neighbors = ScadsEmbeddingVideo.get_related_nodes(target_nodes,
                                                                 limit=self.num_related_class * 10 * ct,
                                                                 only_with_images=processed_embeddings_exist)
                    
                    for neighbor in neighbors:
                        if get_clips(neighbor, all_related_class, True):
                            log.info(f"Neighbor accepted: {neighbor.node} and concept: {conceptnet_id}")
                            cur_related_class += 1
                            all_related_class += 1
                            if cur_related_class >= self.num_related_class:
                                break
                    ct = ct * 2
                    
                    #log.info(f"Curr related: {cur_related_class}, and abs related: {all_related_class}, and iterations: {iters}")
                    iters += 1
        
            Scads.close()
            Cache.set('scads', self.task.classes,
                      (clip_paths, clip_labels, dictionary_clips, all_related_class))

        # Do train/val split
        if split:
            log.info(f"SPLIT TRAIN AND TEST")
            train_percent = 0.8
            num_data = len(clip_paths)
            indices = list(range(num_data))
            train_split = int(np.floor(train_percent * num_data))
            np.random.shuffle(indices)
            train_idx = indices[:train_split]
            val_idx = indices[train_split:]

            train_dataset = HandleExceptionCustomVideoDataset(np.array(clip_paths)[train_idx],
                                                labels=np.array(clip_labels)[train_idx],
                                                transform_img=self.transform_image(video=self.video),
                                                transform_vid=self.transformer_video(),
                                                clips_dictionary=dictionary_clips)

            val_dataset = HandleExceptionCustomVideoDataset(np.array(clip_paths)[val_idx],
                                                 labels=np.array(clip_labels)[val_idx],
                                                 transform_img=self.transform_image(video=self.video),
                                                 transform_vid=self.transformer_video(),
                                                 clips_dictionary=dictionary_clips)
            
            return train_dataset, val_dataset, all_related_class

        else:
            train_dataset = HandleExceptionCustomVideoDataset(clip_paths,
                                                labels=clip_labels,
                                                transform_img=self.transform_image(video=self.video),
                                                transform_vid=self.transformer_video(),
                                                clips_dictionary=dictionary_clips)
        
            return train_dataset, None, all_related_class




class ImageTaglet(ImageTrainable, TagletMixin):
    """
    A trainable model that produces votes for unlabeled images
    """
    
    
class ImageTagletWithAuxData(AuxDataMixin, TagletMixin, ImageTrainable):
    """
    A trainable model that produces votes for unlabeled images and uses ScadsEmbedding to get auxiliary data
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

