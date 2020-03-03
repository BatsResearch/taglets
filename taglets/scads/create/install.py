import argparse
import os
import pandas as pd
from ..create.scads_classes import Node, Image
from ..create.create_scads import add_conceptnet
from ..create.add_datasets import add_dataset

parser = argparse.ArgumentParser(description='Scads')

parser.add_argument("--db", type=str, help="Path to database", required=True)

parser.add_argument("--conceptnet", type=str, help="Path to ConceptNet directory")

parser.add_argument("--cifar100", type=str, help="Path to CIFAR100 directory")

parser.add_argument("--mnist", type=str, help="Path to MNIST directory")

args = parser.parse_args()


class DatasetInstaller:
    def get_name(self):
        raise NotImplementedError()

    def get_images(self, dataset, session):
        raise NotImplementedError()


class CifarInstallation(DatasetInstaller):
    def get_name(self):
        return "CIFAR100"

    def get_images(self, dataset, session):
        all_images = []
        for mode in os.listdir(dataset.path):
            if mode.startswith('.'):
                continue
            mode_dir = os.path.join(dataset.path, mode)
            for label in os.listdir(mode_dir):
                if label.startswith('.'):
                    continue
                node = session.query(Node).filter_by(conceptnet_id=self.get_conceptnet_id(label)).first()
                if not node:
                    continue  # Map is missing a missing conceptnet id
                label_dir = os.path.join(mode_dir, label)
                for image in os.listdir(label_dir):
                    img = Image(dataset_id=dataset.id,
                                node_id=node.id,
                                path=os.path.join(dataset.path, label_dir, image))
                    all_images.append(img)
        return all_images

    def get_conceptnet_id(self, label):
        return "/c/en/" + label


class MnistInstallation:
    def get_name(self):
        return "MNIST"

    def get_images(self, dataset, session):
        modes = ['train', 'test']

        all_images = []
        for mode in modes:
            df_label = pd.read_feather(dataset.path + '/labels_' + mode + '.feather')
            mode_dir = os.path.join(dataset.path, mode)
            for image in os.listdir(mode_dir):
                if image.startswith('.'):
                    continue

                label = df_label.loc[df_label['id'] == image].label.values[0]
                node = session.query(Node).filter_by(conceptnet_id=self.get_conceptnet_id(label)).first()
                if not node:
                    continue  # Map is missing a missing conceptnet id
                img = Image(dataset_id=dataset.id,
                            node_id=node.id,
                            path=os.path.join(dataset.path, mode_dir, image))
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


class Installer:
    def __init__(self, path_to_database):
        self.db = path_to_database

    def install_conceptnet(self, path_to_conceptnet):
        add_conceptnet(self.db, path_to_conceptnet)

    def install_dataset(self, path_to_dataset, dataset_installer):
        add_dataset(self.db, path_to_dataset, dataset_installer)


if __name__ == "__main__":
    installer = Installer(args.db)
    if args.conceptnet:
        installer.install_conceptnet(args.conceptnet)
    if args.cifar100:
        installer.install_dataset(args.cifar100, CifarInstallation())
    if args.mnist:
        installer.install_dataset(args.mnist, MnistInstallation())
