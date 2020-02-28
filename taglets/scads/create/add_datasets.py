import pandas as pd
import os
from sqlalchemy import and_
from .scads_classes import Node, LabelMap, Image


def get_label_map(map_dict, dataset_key, session):
    """
    Get the list of LabelMaps from dictionary of labels.
    :param map_dict: A dictionary from labels to wordnet ids.
        e.g., {'streetcar': 104342573,  'castle': 102983900}
    :param dataset_key: The key of the dataset in the datasets table
    :param session: The current SQLAlchemy session
    :return: A list of 'LabelMap' objects to be inserted into the 'label_maps' dataset
    """
    all_labels = []
    for label_name, wordnet_id in map_dict.items():
        node_results = session.query(Node).filter(Node.name.like('%'+wordnet_id+'%'))
        node_key = None
        for r in node_results:
            node_key = r.key

        label_map = LabelMap(label=label_name, node_key=node_key, dataset_key=dataset_key)
        all_labels.append(label_map)
    return all_labels


def get_images(dataset_key, dataset_name, session):
    """
    Get a list of all images related to a given dataset.
    :param dataset_key: The key of the dataset
    :param dataset_name: The name of the dataset
    :param session: The current SQLAlchemy session
    :return: A list of Images
    """
    if dataset_name == 'CIFAR100':
        return get_cifar100_images(dataset_key, session)
    elif dataset_name == 'MNIST':
        return get_mnist_image(dataset_key, session)


def get_cifar100_images(cifar_key, session):
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
    cwd = os.getcwd()
    cifar100_dir = os.path.join(cwd, 'sql_data', 'CIFAR100')
    all_images = []
    for mode in os.listdir(cifar100_dir):
        if mode.startswith('.'):
            continue
        class_dir = os.path.join(cifar100_dir, mode)
        for label in os.listdir(class_dir):
            if label.startswith('.'):
                continue
            label_map_results = session.query(LabelMap).filter(and_(LabelMap.label == label,
                                                                    LabelMap.dataset_key == cifar_key)).first()
            if label_map_results is None:
                node_key = None
            else:
                node_key = label_map_results.node_key
            image_dir = os.path.join(class_dir, label)
            for image in os.listdir(image_dir):
                img = Image(dataset_key=cifar_key,
                            node_key=node_key,
                            mode=mode,
                            location=os.path.join(image_dir, image))
                all_images.append(img)
    return all_images


def get_mnist_image(dataset_key, session):
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
    cwd = os.getcwd()
    data_dir = os.path.join(cwd, 'sql_data', "MNIST")
    modes = ['train', 'test']

    all_images = []
    for mode in modes:
        if mode.startswith('.') or not os.path.isdir(data_dir):
            continue
        df_label = pd.read_feather(data_dir + '/labels_' + mode + '.feather')
        image_dir = os.path.join(data_dir, mode)
        for image in os.listdir(image_dir):
            if image.startswith('.'):
                continue

            label = df_label.loc[df_label['id'] == image].label.values[0]
            label_map_results = session.query(LabelMap).filter(and_(LabelMap.label == label,
                                                                    LabelMap.dataset_key == dataset_key)).first()
            if label_map_results is None:
                node_key = None
            else:
                node_key = label_map_results.node_key
            img = Image(dataset_key=dataset_key,
                        node_key=node_key,
                        mode=mode,
                        location=os.path.join(image_dir, image))
            all_images.append(img)
    return all_images


def get_map_dict(dataset_name):
    """
    Get a dictionary from labels to wordnet ids.
    :param dataset_name: The name of the dataset
    :return: A dictionary from labels to wordnet ids
    """

    if dataset_name == 'MNIST':
        return {'0': '13742358',
                '1': '13742573',
                '2': '13743269',
                '3': '13744044',
                '4': '13744304',
                '5': '13744521',
                '6': '13744722',
                '7': '13744916',
                '8': '13745086',
                '9': '13745270'}
    elif dataset_name == 'CIFAR100':
        # TODO: Complete this dictionary.
        return {'streetcar': '104342573', 'castle': '102983900', 'bicycle': '102837983', 'motorcycle': '103796045'}


def add_dataset(dataset_info):
    """
    Insert the dataset and its dependencies (LabelMaps, Images) into the database.
    :param dataset_info: A dictionary the dataset name and the number of classes.
    :return: None
    """
    Base.metadata.create_all(engine)
    session = Session()

    dataset_name = dataset_info['dataset_name']
    dataset_nb_classes = dataset_info['nb_classes']

    # Add Dataset to database
    dataset_key = session.query(Dataset.key).filter_by(name=dataset_name).first()

    if dataset_key is not None:
        dataset_key = dataset_key[0]
        dataset = session.query(Dataset).filter_by(key=dataset_key).first()
        print('Dataset already exits.')
    else:
        dataset = Dataset(name=dataset_name, nb_classes=dataset_nb_classes)
        session.add(dataset)
        session.commit()
        print(dataset_name, 'added to datasets table.')

    # Add LabelMaps to database
    map_dict = get_map_dict(dataset_name)

    all_label_maps = get_label_map(map_dict, dataset_key, session)

    for label_map in all_label_maps:
        label_map.dataset = dataset
        if label_map.node_key is not None:
            label_map.node = session.query(Node).filter(Node.key == label_map.node_key).first()

    session.add_all(all_label_maps)
    session.commit()
    print("LabelMaps related to", dataset_name, "added to label_maps dataset.")

    # Add Images to database
    all_images = get_images(dataset_key, dataset_name, session)

    for image in all_images:
        image.dataset = dataset
        if image.node_key is not None:
            image.node = session.query(Node).filter(Node.key == image.node_key).first()
    session.add_all(all_images)
    session.commit()
    print("Images in", dataset_name, "added to images dataset.")


def add_datasets():
    """
    Add MNIST and CIFAR100 to the database.
    :return: None
    """
    mnist_info = {'dataset_name': 'MNIST',
                  'nb_classes': 10}
    add_dataset(mnist_info)

    cifar100_info = {'dataset_name': 'CIFAR100',
                     'nb_classes': 100}
    add_dataset(cifar100_info)
