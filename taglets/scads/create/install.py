import argparse
import os
import pandas as pd
from ..create.scads_classes import Node, Image, Clip
from ..create.create_scads import add_conceptnet
from ..create.add_datasets import add_dataset
from .wnids_to_concept import SYNSET_TO_CONCEPTNET_ID
import imagesize
import numpy as np
import re

class DatasetInstaller:
    def get_name(self):
        raise NotImplementedError()

    def get_data(self, dataset, session, root):
        raise NotImplementedError()
    
    def get_conceptnet_id(self, label):
        return "/c/en/" + label.lower().replace(" ", "_").replace("-", "_")


class ImageClassificationInstaller(DatasetInstaller):
    def get_name(self):
        raise NotImplementedError()

    def get_data(self, dataset, session, root):
        size = "full"
        modes = ['train', 'test']
        label_to_node_id = {}
        
        all_images = []
        for mode in modes:
            df_label = pd.read_feather(
                os.path.join(root, dataset.path, "labels_" + size, 'labels_' + mode + '.feather'))
            df = pd.crosstab(df_label['id'], df_label['class'])
            mode_dir = os.path.join(dataset.path, f'{dataset.path}_' + size, mode)
            for image in os.listdir(os.path.join(root, mode_dir)):
                if image.startswith('.'):
                    continue
                
                label = df.loc[image].idxmax()
                # Get node_id
                if label in label_to_node_id:
                    node_id = label_to_node_id[label]
                else:
                    node = session.query(Node).filter_by(conceptnet_id=self.get_conceptnet_id(label)).first()
                    node_id = node.id if node else None
                    label_to_node_id[label] = node_id
                
                # Scads is missing a missing conceptnet id
                if not node_id:
                    continue
                
                img = Image(dataset_id=dataset.id,
                            node_id=node_id,
                            path=os.path.join(mode_dir, image))
                all_images.append(img)
        return all_images
    
    
class ObjectDetectionInstaller(DatasetInstaller):
    def get_name(self):
        raise NotImplementedError()

    def get_data(self, dataset, session, root):
        size = "full"
        modes = ['train', 'test']
        label_to_node_id = {}
        
        all_images = []
        for mode in modes:
            df_label = pd.read_feather(
                os.path.join(root, dataset.path, "labels_" + size, 'labels_' + mode + '.feather'))
            bbox = list(df_label.loc[:, 'bbox'].copy())
            for i in range(len(bbox)):
                bbx = bbox[i].split(',')
                x_min = float(bbx[0].strip())
                y_min = float(bbx[1].strip())
                x_max = float(bbx[2].strip())
                y_max = float(bbx[3].strip())
                w = x_max - x_min
                h = y_max - y_min
                area = w * h
                bbox[i] = area
            df_label.loc[:, 'bbox'] = bbox
            pt = df_label.pivot_table(index='id', columns='class', values='bbox', aggfunc=np.max)
            mode_dir = os.path.join(dataset.path, f'{dataset.path}_' + size, mode)
            for image in os.listdir(os.path.join(root, mode_dir)):
                if image.startswith('.'):
                    continue
                
                width, height = imagesize.get(os.path.join(root, mode_dir) + '/' + image)
                img_size = width * height
                label = pt.loc[image].dropna().idxmax()
                bbox_area = pt.loc[image, label]
                if bbox_area <= img_size * .2:
                    continue
                # Get node_id
                if label in label_to_node_id:
                    node_id = label_to_node_id[label]
                else:
                    node = session.query(Node).filter_by(conceptnet_id=self.get_conceptnet_id(label)).first()
                    node_id = node.id if node else None
                    label_to_node_id[label] = node_id
                
                # Scads is missing a missing conceptnet id
                if not node_id:
                    continue
                
                img = Image(dataset_id=dataset.id,
                            node_id=node_id,
                            path=os.path.join(mode_dir, image))
                all_images.append(img)
        return all_images


class VideoClassificationInstaller(DatasetInstaller):
    def get_name(self):
        raise NotImplementedError()

    def composed_labels(self, string, dataset):
        if dataset.name == 'HMDB':
            return [w.strip() for w in string.split('_')]
        elif dataset.name == 'UCF101':
            list_words = re.findall('[A-Z][^A-Z]*', string)
            return [w.lower().strip() for w in list_words]
        elif dataset.name == 'Kinetics400':
            return [w.strip() for w in string.split('/')[-1].split('_') if len(w) > 2]

    
    def get_data(self, dataset, session, root):
        size = "full"
        modes = ['train', 'test']
        label_to_node_id = {}

        all_clips = []
        for mode in modes:
            base_path = os.path.join(dataset.path, dataset.path + '_' + size, mode)

            #print(base_path, dataset.path, root)
            df = pd.read_feather(
                os.path.join(root, dataset.path, "labels_" + size, 'labels_' + mode + '.feather'))
            if mode == "test":
                df_label = pd.crosstab(df['id'], df['class'])
                df = pd.read_feather(
                    os.path.join(root, dataset.path, "labels_" + size, "meta_" + mode + ".feather")
                )

            for _, row in df.iterrows():
                row = row.astype("object")
                if mode == "test":
                    label = df_label.loc[row['id']].idxmax()
                else:
                    label = row['class']
                
                if label in label_to_node_id:
                    node_id = label_to_node_id[label]
                    print(os.path.join(base_path, str(row['video_id'])))
                    clip = Clip(
                            clip_id=row['id'],
                            video_id=row['video_id'],
                            base_path=os.path.join(base_path, str(row['video_id'])),
                            start_frame=row['start_frame'],
                            end_frame=row['end_frame'],
                            real_label=self.get_conceptnet_id(label).split('/')[-1],
                            dataset_id=dataset.id,
                            node_id=node_id
                            )
                    all_clips.append(clip)

                else: 
                    node = session.query(Node).filter_by(conceptnet_id=self.get_conceptnet_id(label)).first()
                    # If the node related to the class doesn't exist
                    if node:
                        node_id = node.id 
                        label_to_node_id[label] = node_id
                        clip = Clip(
                                    clip_id=row['id'],
                                    video_id=row['video_id'],
                                    base_path=os.path.join(base_path, str(row['video_id'])),
                                    start_frame=row['start_frame'],
                                    end_frame=row['end_frame'],
                                    real_label=self.get_conceptnet_id(label).split('/')[-1],
                                    dataset_id=dataset.id,
                                    node_id=node_id
                                )
                        all_clips.append(clip)
                    # Else, we decompose the class name and check for concepts corresponding to each word
                    else:
                        labels = self.composed_labels(label, dataset)

                        for l in labels:
                            if l in label_to_node_id:
                                node_id = label_to_node_id[l]
                            else: 
                                node = session.query(Node).filter_by(conceptnet_id=self.get_conceptnet_id(l)).first()
                                node_id = node.id if node else None
                                label_to_node_id[l] = node_id

                            # Handle the case when a classe, even decomposed, is not assign to any concept
                            if not node_id:
                                continue
                            
                            real = '_'.join(labels)
                            clip = Clip(
                                clip_id=row['id'],
                                video_id=row['video_id'],
                                base_path=os.path.join(base_path, str(row['video_id'])),
                                start_frame=row['start_frame'],
                                end_frame=row['end_frame'],
                                real_label=self.get_conceptnet_id(real).split('/')[-1],
                                dataset_id=dataset.id,
                                node_id=node_id
                            )
                            all_clips.append(clip)
                                           
        return all_clips


class CifarInstallation(ImageClassificationInstaller):
    def get_name(self):
        return "CIFAR100"


class MnistInstallation(ImageClassificationInstaller):
    def get_name(self):
        return "MNIST"

    def get_conceptnet_id(self, label):
        mnist_classes = {
            '0': '/c/en/zero',
            '1': '/c/en/one',
            '2': '/c/en/two',
            '3': '/c/en/three',
            '4': '/c/en/four',
            '5': '/c/en/five',
            '6': '/c/en/six',
            '7': '/c/en/seven',
            '8': '/c/en/eight',
            '9': '/c/en/nine',
        }
        return mnist_classes[label]


class ImageNetInstallation(ImageClassificationInstaller):
    def get_name(self):
        return "ImageNet"

    def get_conceptnet_id(self, label):
        return SYNSET_TO_CONCEPTNET_ID[label]


class COCO2014Installation(ObjectDetectionInstaller):
    def get_name(self):
        return "COCO2014"

    def get_conceptnet_id(self, label):
        label_to_label = {1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train',
                          8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant', 12: '-', 13: 'stop sign',
                          14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep',
                          21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 26: '-', 27: 'backpack',
                          28: 'umbrella', 29: '-', 30: '-', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee',
                          35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat',
                          40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle',
                          45: '-', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl',
                          52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot',
                          58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch',
                          64: 'potted plant', 65: 'bed', 66: '-', 67: 'dining table', 68: '-', 69: '-', 70: 'toilet',
                          71: '-', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone',
                          78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator', 83: '-',
                          84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier',
                          90: 'toothbrush', 91: '-', 92: ''}
        return "/c/en/" + label_to_label[label].lower().replace(" ", "_").replace("-", "_")


class DomainNetInstallation(ImageClassificationInstaller):
    def __init__(self, domain_name):
        self.domain = domain_name

    def get_name(self):
        return "DomainNet: " + self.domain

    def get_conceptnet_id(self, label):
        exceptions = {'paint_can': 'can_of_paint',
                      'The_Eiffel_Tower': 'eiffel_tower',
                      'animal_migration': 'migration',
                      'teddy-bear': 'teddy_bear',
                      'The_Mona_Lisa': 'mona_lisa',
                      't-shirt': 't_shirt',
                      'The_Great_Wall_of_China': 'great_wall_of_china'}
        if label in exceptions:
            return "/c/en/" + exceptions[label]
        return "/c/en/" + label.lower().replace(" ", "_").replace("-", "_")

class MslCuriosityInstallation(ImageClassificationInstaller):
    def get_name(self):
        return "MslCuriosity"

    def get_conceptnet_id(self, label):
        
        exceptions = {'drill holes' : 'holes',
                     'observation tray' : 'tray',
                     'rover rear deck' : 'deck'}
        if label in exceptions:
            return "/c/en/" + exceptions[label]
        return "/c/en/" + label.lower().replace(" ", "_").replace("-", "_")
    
class MarsSurfaceInstallation(ImageClassificationInstaller):
    def get_name(self):
        return "MarsSurface"

    def get_conceptnet_id(self, label):
        
        exceptions = {'drill holes' : 'holes',
                     'observation tray' : 'tray',
                     'rover rear deck' : 'deck'}
        if label in exceptions:
            return "/c/en/" + exceptions[label]
        return "/c/en/" + label.lower().replace(" ", "_").replace("-", "_")


class VOC2009Installation(ObjectDetectionInstaller):
    def get_name(self):
        return "VOC2009"

    def get_conceptnet_id(self, label):
        exceptions = {'pottedplant': 'potted_plant',
                      'tvmonitor': 'tv_monitor',
                      'diningtable': 'dining_table'}
        if label in exceptions:
            return "/c/en/" + exceptions[label]
        return "/c/en/" + label.lower().replace(" ", "_").replace("-", "_")


class GoogleOpenImageInstallation(ObjectDetectionInstaller):
    def get_name(self):
        return "GoogleOpenImage"


class HMDBInstallation(VideoClassificationInstaller):
    def get_name(self):
        return "HMDB"

class UCF101Installation(VideoClassificationInstaller):
    def get_name(self):
        return "UCF101"
    def get_conceptnet_id(self, label):
        exceptions = {'Skijet': 'jet_ski'}
        if label in exceptions:
            return "/c/en/" + exceptions[label]
        else:
            label_clean = '_'.join([i.lower() for i in re.findall('[A-Z][^A-Z]*', label)])
            if len(label_clean) != 0:
                return "/c/en/" + label_clean#label.lower().replace(" ", "_")#.replace("-", "_")
            else:
                return "/c/en/" + label.lower()
            
class Kinetics400Installation(VideoClassificationInstaller):
    def get_name(self):
        return "Kinetics400"
    def get_conceptnet_id(self, label):
        exceptions = {'water_sliding': 'water_slide',
              'playing_flute': 'play_flute',
              'skiing_slalom': 'ski_slalom',
              'bending_metal': 'bend_metal',
              'dying_hair' : 'dye_hair',
              'playing_recorder' : 'play_recorder',
              'cooking_egg' : 'cook_egg',
              'eating_watermelon': 'eat_watermelon',
              'opening_bottle':'open_bottle',
              'news_anchoring' : 'news_anchor'
             }
        if label in exceptions:
            return "/c/en/" + exceptions[label]
        else:
            label_clean = label.replace('(','').replace(')','').replace("''",'').lower()
            if len(label_clean) != 0:
                return "/c/en/" + label_clean
            else:
                return "/c/en/" + label.lower()


class Installer:
    def __init__(self, path_to_database):
        self.db = path_to_database

    def install_conceptnet(self, path_to_conceptnet):
        add_conceptnet(self.db, path_to_conceptnet)

    def install_dataset(self, root, path_to_dataset, dataset_installer):
        add_dataset(self.db, root, path_to_dataset, dataset_installer)


if __name__ == "__main__":
    # Get arguments
    parser = argparse.ArgumentParser(description='Scads')
    parser.add_argument("--db", type=str, help="Path to database", required=True)
    parser.add_argument("--conceptnet", type=str, help="Path to ConceptNet directory")
    parser.add_argument("--root", type=str, help="Root containing dataset directories")
    parser.add_argument("--cifar100", type=str, help="Path to CIFAR100 directory from the root")
    parser.add_argument("--mnist", type=str, help="Path to MNIST directory from the root")
    parser.add_argument("--imagenet", type=str, help="Path to ImageNet directory from the root")
    parser.add_argument("--coco2014", type=str, help="Path to COCO2014 directory from the root")
    parser.add_argument("--voc2009", type=str, help="Path to voc2009 directory from the root")
    parser.add_argument("--googleopenimage", type=str, help="Path to googleopenimage directory from the root")
    parser.add_argument("--domainnet", nargs="+")
    parser.add_argument("--hmdb", type=str, help="Path to hmdb directory from the root")
    parser.add_argument("--ucf101", type=str, help="Path to ufc101 directory from the root")
    parser.add_argument("--kinetics400", type=str, help="Path to kinetics directory from the root")
    parser.add_argument("--msl_curiosity", type=str, help="Path to msl_curiosity directory from the root")
    parser.add_argument("--mars_surface_imgs", type=str, help="Path to mars_surface_imgs directory from the root")
    
    args = parser.parse_args()

    # Install SCADS
    installer = Installer(args.db)
    if args.conceptnet:
        installer.install_conceptnet(args.conceptnet)
    if args.cifar100:
        if not args.root:
            raise RuntimeError("Must specify root directory.")
        installer.install_dataset(args.root, args.cifar100, CifarInstallation())
    if args.mnist:
        if not args.root:
            raise RuntimeError("Must specify root directory.")
        installer.install_dataset(args.root, args.mnist, MnistInstallation())
    if args.imagenet:
        if not args.root:
            raise RuntimeError("Must specify root directory.")
        installer.install_dataset(args.root, args.imagenet, ImageNetInstallation())
    if args.coco2014:
        if not args.root:
            raise RuntimeError("Must specify root directory.")
        installer.install_dataset(args.root, args.coco2014, COCO2014Installation())

    if args.voc2009:
        if not args.root:
            raise RuntimeError("Must specify root directory.")
        installer.install_dataset(args.root, args.voc2009, VOC2009Installation())

    if args.googleopenimage:
        if not args.root:
            raise RuntimeError("Must specify root directory.")
        installer.install_dataset(args.root, args.googleopenimage, GoogleOpenImageInstallation())

    if args.domainnet:
        if not args.root:
            raise RuntimeError("Must specify root directory.")
        for domain in args.domainnet:
            name = domain.split("-")[1].capitalize()
            installer.install_dataset(args.root, domain, DomainNetInstallation(name))

    if args.hmdb:
        if not args.root:
            raise RuntimeError("Must specify root directory.")
        installer.install_dataset(args.root, args.hmdb, HMDBInstallation())

    if args.ucf101:
        if not args.root:
            raise RuntimeError("Must specify root directory.")
        installer.install_dataset(args.root, args.ucf101, UCF101Installation())
        
    if args.kinetics:
        if not args.root:
            raise RuntimeError("Must specify root directory.")
        installer.install_dataset(args.root, args.kinetics400, Kinetics400Installation())

    if args.msl_curiosity:
        if not args.root:
            raise RuntimeError("Must specify root directory.")
        installer.install_dataset(args.root, args.msl_curiosity, MslCuriosityInstallation())

    if args.mars_surface_imgs:
        if not args.root:
            raise RuntimeError("Must specify root directory.")
        installer.install_dataset(args.root, args.mars_surface_imgs, MarsSurfaceInstallation())
    
