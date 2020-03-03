import pandas as pd
import os
from sqlalchemy import and_
from .scads_classes import Node, Image


def get_images(dataset, session):
    """
    Get a list of all images related to a given dataset.
    :param dataset_key: The key of the dataset
    :param dataset_name: The name of the dataset
    :param session: The current SQLAlchemy session
    :return: A list of Images
    """
    if dataset.name == 'CIFAR100':
        return get_cifar100_images(dataset, session)
    elif dataset.name == 'MNIST':
        return get_mnist_image(dataset, session)


def get_cifar100_images(dataset, session):
    """
    Get all CIFAR Images.

    For each image in cifar dataset, create an object of 'Image', e.g.,:
        apple_img_1 = Image(dataset_key='cifar100 key',
                            node_key='apple key',
                            mode = 'train',
                            location='directory/apple_1.png')

    Images in CIFAR100 are split into train and test.
        /CIFAR100:
            /train
                /fish
                /bicycle
                /beaver
                ...
            /test
                /fish
                /bicycle
                /beaver
                ...

    :param cifar_key: The key of CIFAR in the datasets table
    :param session: The current SQLAlchemy session.
    :return: A list of Images
    """
    all_images = []
    for mode in os.listdir(dataset.path):
        if mode.startswith('.'):
            continue
        mode_dir = os.path.join(dataset.path, mode)
        for label in os.listdir(mode_dir):
            if label.startswith('.'):
                continue
            node = session.query(Node).filter_by(conceptnet_id=get_conceptnet_id(dataset.name, label)).first()
            if not node:
                continue    # Map is missing a missing conceptnet id
            label_dir = os.path.join(mode_dir, label)
            for image in os.listdir(label_dir):
                img = Image(dataset_id=dataset.id,
                            node_id=node.id,
                            path=os.path.join(dataset.path, label_dir, image))
                all_images.append(img)
    return all_images


def get_mnist_image(dataset, session):
    """
    Get all MNIST Images.

    For each image in MNIST dataset, create an object of 'Image', e.g.,:
        one_img_1 = Image(dataset_key='MNIST key',
                          node_key='one key',
                          mode = 'train',
                          location='directory/one_1.png')

    Images in MNIST are split into train and test.
        /MNIST:
            /train
                /1.png
                /2.png
                /3.png
                ...
            /test
                /1.png
                /2.png
                /3.png
                ...

    There are two label files in dataset directory: one for train and one for test.

    :param dataset_key: The key in the datasets table
    :param session: The current SQLAlchemy session
    :return: A list of Images
    """
    modes = ['train', 'test']

    all_images = []
    for mode in modes:
        df_label = pd.read_feather(dataset.path + '/labels_' + mode + '.feather')
        mode_dir = os.path.join(dataset.path, mode)
        for image in os.listdir(mode_dir):
            if image.startswith('.'):
                continue

            label = df_label.loc[df_label['id'] == image].label.values[0]
            node = session.query(Node).filter_by(conceptnet_id=get_conceptnet_id(dataset.name, label)).first()
            if not node:
                continue  # Map is missing a missing conceptnet id
            img = Image(dataset_id=dataset.id,
                        node_id=node.id,
                        path=os.path.join(dataset.path, mode_dir, image))
            all_images.append(img)
    return all_images


def get_conceptnet_id(dataset_name, label):
    """
    Get a dictionary from labels to conceptnet ids.
    :param dataset_name: The name of the dataset
    :return: A dictionary from labels to conceptnetnet ids
    """
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
    if dataset_name == "CIFAR100":
        return "/c/en/" + label
    elif dataset_name == "MNIST":
        return mnist_classes[label]
    else:
        return None


def add_dataset(dataset_name, path):
    """
    Insert the dataset and its dependencies (LabelMaps, Images) into the database.
    :param dataset_info: A dictionary the dataset name and the number of classes.
    :return: None
    """
    if not os.path.isdir(path):
        print("Path does not exist.")

    # Get session
    session = Session()

    # Add Dataset to database
    dataset = session.query(Dataset).filter_by(name=dataset_name).first()
    if not dataset:
        dataset = Dataset(name=dataset_name, path=path)
        session.add(dataset)
        session.commit()
        print(dataset_name, 'added to datasets table.')
    else:
        print('Dataset already exits at', dataset.path)
        return

    # Add Images to database
    all_images = get_images(dataset, session)
    for image in all_images:
        image.dataset = dataset
        if image.node_id:
            image.node = session.query(Node).filter(Node.id == image.node_id).one()
    session.add_all(all_images)
    session.commit()
    print("Images in", dataset_name, "added to images dataset.")


def add_datasets():
    """
    Add MNIST and CIFAR100 to the database.
    :return: None
    """
    add_dataset('MNIST', '../../test_data/MNIST')
    add_dataset('CIFAR100', '../../test_data/CIFAR100')


def main():
    add_datasets()


if __name__ == "__main__":
    main()
