import argparse
import os
import pandas as pd
from ..create.scads_classes import Node, Image
from ..create.create_scads import add_conceptnet
from ..create.add_datasets import add_dataset
import requests
import time


class DatasetInstaller:
    def get_name(self):
        raise NotImplementedError()

    def get_images(self, dataset, session, root):
        raise NotImplementedError()

class CifarInstallation(DatasetInstaller):
    def get_name(self):
        return "CIFAR100"

    def get_images(self, dataset, session, root):
        size = "full"
        modes = ['train', 'test']
        label_to_node_id = {}

        all_images = []
        for mode in modes:
            df_label = pd.read_feather(
                os.path.join(root, dataset.path, "labels_" + size, 'labels_' + mode + '.feather'))
            mode_dir = os.path.join(dataset.path, "cifar100_" + size, mode)
            for image in os.listdir(os.path.join(root, mode_dir)):
                if image.startswith('.'):
                    continue
                label = df_label.loc[df_label['id'] == image]['class'].values[0]

                # Get node_id
                if label in label_to_node_id:
                    node_id = label_to_node_id[label]
                else:
                    node = session.query(Node).filter_by(conceptnet_id=self.get_conceptnet_id(label)).first()
                    node_id = node.id if node else None
                    label_to_node_id[label] = node_id
                if not node_id:
                    continue  # Scads is missing a missing conceptnet id
                img = Image(dataset_id=dataset.id,
                            node_id=node_id,
                            path=os.path.join(mode_dir, image))
                all_images.append(img)
        return all_images

    def get_conceptnet_id(self, label):
        return "/c/en/" + label.lower().replace(" ", "_")

class MnistInstallation(DatasetInstaller):
    def get_name(self):
        return "MNIST"

    def get_images(self, dataset, session, root):
        size = "full"
        modes = ['train', 'test']
        label_to_node_id = {}

        all_images = []
        for mode in modes:
            df_label = pd.read_feather(
                os.path.join(root, dataset.path, "labels_" + size, 'labels_' + mode + '.feather'))
            mode_dir = os.path.join(dataset.path, "mnist_" + size, mode)
            for image in os.listdir(os.path.join(root, mode_dir)):
                if image.startswith('.'):
                    continue
                label = df_label.loc[df_label['id'] == image]['class'].values[0]

                # Get node_id
                if label in label_to_node_id:
                    node_id = label_to_node_id[label]
                else:
                    node = session.query(Node).filter_by(conceptnet_id=self.get_conceptnet_id(label)).first()
                    node_id = node.id if node else None
                    label_to_node_id[label] = node_id
                if not node_id:
                    continue  # Scads is missing a missing conceptnet id
                img = Image(dataset_id=dataset.id,
                            node_id=node_id,
                            path=os.path.join(mode_dir, image))
                all_images.append(img)
        return all_images

    def get_conceptnet_id(self, label):
        mnist_classes = {
            '0': '/c/en/zero/n/wn/quantity',
            '1': '/c/en/one/n/wn/quantity',
            '2': '/c/en/two/n/wn/quantity',
            '3': '/c/en/three/n/wn/quantity',
            '4': '/c/en/four/n/wn/quantity',
            '5': '/c/en/five/n/wn/quantity',
            '6': '/c/en/six/n/wn/quantity',
            '7': '/c/en/seven/n/wn/quantity',
            '8': '/c/en/eight/n/wn/quantity',
            '9': '/c/en/nine/n/wn/quantity',
        }
        return mnist_classes[label]

class ImageNetInstallation(DatasetInstaller):
    def get_name(self):
        return "ImageNet"

    def get_images(self, dataset, session, root):
        size = "full"
        modes = ['train', 'test']
        synset_to_labels_endpoint = "http://www.image-net.org/api/text/wordnet.synset.getwords?wnid="
        synset_to_labels = {}
        label_to_node_id = {}

        all_images = []
        image_counter = 0
        missed_labeles = set([])
        for mode in modes:
            counter = 0
            df_label = pd.read_feather(
                os.path.join(root, dataset.path, "labels_" + size, 'labels_' + mode + '.feather'))
            mode_dir = os.path.join(dataset.path, "imagenet_1k_" + size, mode)
            for image in os.listdir(os.path.join(root, mode_dir)):
                if image.startswith('.'):
                    continue

                # Get labels
                synset = df_label.loc[df_label['id'] == image]['class'].values[0]
                if synset in synset_to_labels:
                    labels = synset_to_labels[synset]
                else:

                    try:
                        labels = requests.get(synset_to_labels_endpoint + synset)
                    except:
                        print("Connection refused by the server..")
                        print("Let me sleep for 60 seconds")
                        print("ZZzzzz...")
                        time.sleep(60)
                        print("Was a nice sleep, now let me continue...")
                        labels = requests.get(synset_to_labels_endpoint + synset)
                        continue
                    labels = list(filter(lambda x: x, labels.text.split("\n")))
                    synset_to_labels[synset] = labels

                # Get nodes
                for label in labels:
                    # Get node_id
                    if label in label_to_node_id:
                        node_id = label_to_node_id[label]
                    else:
                        node = session.query(Node).filter_by(conceptnet_id=self.get_conceptnet_id(label)).first()
                        node_id = node.id if node else None
                        label_to_node_id[label] = node_id
                    if not node_id:
                        missed_labeles.add(label)
                        continue  # Scads is missing a missing conceptnet id
                    img = Image(dataset_id=dataset.id,
                                node_id=node_id,
                                path=os.path.join(mode_dir, image))
                    all_images.append(img)
                    image_counter += 1
                    if image_counter % 5000 == 0:
                        session.add_all(all_images)
                        session.commit()
                        all_images = []
                        image_counter = 0
                        print('a chunk of 5000 images from imagenet is added to images dataset')
        print('missed_labeles')
        print(missed_labeles)
        return all_images

    def get_conceptnet_id(self, label):
        return "/c/en/" + label.lower().replace(" ", "_")

class COCO2014Installation(DatasetInstaller):
    def get_name(self):
        return "COCO2014"

    def get_images(self, dataset, session, root):
        size = "full"
        modes = ['train', 'test']
        class_to_label = {1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train',
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
        label_to_node_id = {}

        all_images = []
        for mode in modes:
            df_label = pd.read_feather(
                os.path.join(root, dataset.path, "labels_" + size, 'labels_' + mode + '.feather'))
            mode_dir = os.path.join(dataset.path, "coco2014_" + size, mode)
            for image in os.listdir(os.path.join(root, mode_dir)):
                if image.startswith('.'):
                    continue
                label = class_to_label[df_label.loc[df_label['id'] == image]['class'].values[0]]

                # Get node_id
                if label in label_to_node_id:
                    node_id = label_to_node_id[label]
                else:
                    node = session.query(Node).filter_by(conceptnet_id=self.get_conceptnet_id(label)).first()
                    node_id = node.id if node else None
                    label_to_node_id[label] = node_id
                if not node_id:
                    continue  # Scads is missing a missing conceptnet id
                img = Image(dataset_id=dataset.id,
                            node_id=node_id,
                            path=os.path.join(mode_dir, image))
                all_images.append(img)
        return all_images

    def get_conceptnet_id(self, label):
        return "/c/en/" + label.lower().replace(" ", "_")

class DomainNetInstallation(DatasetInstaller):
    def __init__(self, domain_name):
        self.domain = domain_name

    def get_name(self):
        return "DomainNet: " + self.domain

    def get_images(self, dataset, session, root):
        size = "full"
        modes = ['train', 'test']
        label_to_node_id = {}

        all_images = []
        missed_labels = set([])
        for mode in modes:
            df_label = pd.read_feather(
                os.path.join(root, dataset.path, "labels_" + size, 'labels_' + mode + '.feather'))
            mode_dir = os.path.join(dataset.path, dataset.path + "_" + size, mode)
            for image in os.listdir(os.path.join(root, mode_dir)):
                if image.startswith('.'):
                    continue
                label = df_label.loc[df_label['id'] == image]['class'].values[0]

                # Get node_id
                if label in label_to_node_id:
                    node_id = label_to_node_id[label]
                else:
                    node = session.query(Node).filter_by(conceptnet_id=self.get_conceptnet_id(label)).first()
                    node_id = node.id if node else None
                    label_to_node_id[label] = node_id
                if not node_id:
                    missed_labels.add(label)
                    continue  # Scads is missing a missing conceptnet id
                img = Image(dataset_id=dataset.id,
                            node_id=node_id,
                            path=os.path.join(mode_dir, image))
                all_images.append(img)
        print(missed_labels)
        return all_images

    def get_conceptnet_id(self, label):
        exceptions = {'paint_can':'can_of_paint', 'The_Eiffel_Tower':'eiffel_tower', 'animal_migration':'migration', 'teddy-bear':'teddy_bear', 'The_Mona_Lisa':'mona_lisa', 't-shirt':'t_shirt',
         'The_Great_Wall_of_China':'great_wall_of_china'}
        if label in exceptions:
            return "/c/en/" +exceptions[label]
        return "/c/en/" + label.lower().replace(" ", "_")

class VOC2009Installation(DatasetInstaller):
    def get_name(self):
        return "VOC2009"

    def get_images(self, dataset, session, root):
        size = "full"
        modes = ['train', 'test']
        label_to_node_id = {}

        all_images = []
        missed_labels = set([])
        for mode in modes:
            df_label = pd.read_feather(
                os.path.join(root, dataset.path, "labels_" + size, 'labels_' + mode + '.feather'))
            mode_dir = os.path.join(dataset.path, "voc2009_" + size, mode)
            for image in os.listdir(os.path.join(root, mode_dir)):
                if image.startswith('.'):
                    continue
                label = df_label.loc[df_label['id'] == image]['class'].values[0]

                # Get node_id
                if label in label_to_node_id:
                    node_id = label_to_node_id[label]
                else:
                    node = session.query(Node).filter_by(conceptnet_id=self.get_conceptnet_id(label)).first()
                    node_id = node.id if node else None
                    label_to_node_id[label] = node_id
                if not node_id:
                    missed_labels.add(label)
                    continue  # Scads is missing a missing conceptnet id
                img = Image(dataset_id=dataset.id,
                            node_id=node_id,
                            path=os.path.join(mode_dir, image))
                all_images.append(img)
        print('missed_labels:')
        print(missed_labels)
        return all_images

    def get_conceptnet_id(self, label):
        exceptions = {'pottedplant':'potted_plant', 'tvmonitor':'tv_monitor', 'diningtable':'dining_table'}
        if label in exceptions:
            return "/c/en/" +exceptions[label]
        return "/c/en/" + label.lower().replace(" ", "_")

class GoogleOpenImageInstallation(DatasetInstaller):
    def get_name(self):
        return "GoogleOpenImage"

    def get_images(self, dataset, session, root):
        size = "full"
        modes = ['train', 'test']
        label_to_node_id = {}
        missed_labeles=set([])
        all_images = []
        image_counter = 0
        for mode in modes:
            df_label = pd.read_feather(
                os.path.join(root, dataset.path, "labels_" + size, 'labels_' + mode + '.feather'))
            mode_dir = os.path.join(dataset.path, "google_open_image_" + size, mode)
            for image in os.listdir(os.path.join(root, mode_dir)):
                if image.startswith('.'):
                    continue
                label = df_label.loc[df_label['id'] == image]['class'].values[0]

                # Get node_id
                if label in label_to_node_id:
                    node_id = label_to_node_id[label]
                else:
                    node = session.query(Node).filter_by(conceptnet_id=self.get_conceptnet_id(label)).first()
                    node_id = node.id if node else None
                    label_to_node_id[label] = node_id
                if not node_id:
                    missed_labeles.add(label)
                    continue  # Scads is missing a missing conceptnet id
                img = Image(dataset_id=dataset.id,
                            node_id=node_id,
                            path=os.path.join(mode_dir, image))
                all_images.append(img)
                image_counter += 1
                if image_counter % 500 == 0:
                    print(image_counter)
                    session.add_all(all_images)
                    session.commit()
                    all_images = []
                    image_counter = 0
                    print('a chunk of 500 images from google open image is added to images dataset')
        print(missed_labeles)
        return all_images

    def get_conceptnet_id(self, label):
        return "/c/en/" + label.lower().replace(" ", "_")

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