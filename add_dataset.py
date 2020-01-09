from sqlalchemy_scads import *
import pandas as pd

Base = declarative_base()
engine = sa.create_engine('sqlite:///sql_data/scads_db.db')
Session = sessionmaker(bind=engine)


def get_label_map(map_dict, dataset_key, session):
    """load label name and corresponding wordnet id from map_dict, return a list of 'Label_map' class

    :param map_dict: map dictionary; key is label and value is wordnet id. e.g., {'streetcar':104342573,  'castle': 102983900}
    :param dataset_key: key of dataset in 'datasets' table
    :param session: current  sqlalchemy session
    :return: a list of 'Label_map' object that should be inserted into 'label_maps' dataset

    """

    all_labels = []
    for label_name, wordnet_id in map_dict.items():
        node_results = session.query(Node).filter(Node.name.like('%'+wordnet_id+'%'))
        node_key = 99999
        for r in node_results:
            node_key = r.key

        label_map = LabelMap(label=label_name, node_key=node_key, dataset_key=dataset_key)
        all_labels.append(label_map)
    return all_labels


def get_image(dataset_key, dataset_name, session):
    """for the given dataset, call the function to return a list of all images related to the dataset.

    :param dataset_key: key of dataset we want to retrieve information; could be 0, 1, 2, etc.
    :param dataset_name: name of dataset we want to retrieve information; could be 0, 1, 2,
    :param session: current session
    :return:
    """
    if dataset_name == 'CIFAR100':
        return get_cifar100_images(dataset_key, session)
    elif dataset_name == 'MNIST':
        return get_MNIST_image(dataset_key, session)


def get_cifar100_images(cifar_key, session):
    """go over all cifar images in cifar directory, and return a list of 'Image' class.

    For each image in cifar dataset, create an object of 'Image', e.g.,:
        apple_img_1 = Image(dataset_key='cifar100 key', node_key='apple key', mode = 'train', location='directory/apple_1.png')

    Images in cifar100 split to train and test.
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

    'mode' is 'train' or 'test' or 'none' (image in dataset is in 'test' folder or 'train' folder;
    otherwise it should be 'none').

    :param cifar_key: key of cifar in datasets table.
    :param session: current sqlalchemy session.
    :return: a list of Image class for all of the images in cifar.

    """

    cwd = os.getcwd()
    cifar100_dir = os.path.join(cwd, 'sql_data', 'CIFAR100')
    all_images = []
    for mode in os.listdir(cifar100_dir):
        if mode.startswith('.'):
            continue
        class_dir = os.path.join(cifar100_dir,mode)
        for label in os.listdir(class_dir):
            if label.startswith('.'):
                continue
            label_map_results = session.query(LabelMap).filter(LabelMap.label.like('%' + label + '%')).first()
            if label_map_results is None:
                node_key = 9999
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


def get_MNIST_image(dataset_key, session):
    """go over all images in dataset directory, and return a list of 'Image' class.

    For each image in MNIST dataset, create an object of 'Image', e.g.,:
        one_img_1 = Image(dataset_key='MNIST key', node_key='one key', mode = 'train', location='directory/one_1.png')

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

    'mode' is 'train' or 'test' or 'none' (image in dataset is in 'test' folder or 'train' folder;
    otherwise it should be 'none').

    there are two label files in dataset directory, one for trian and the other is for test.

    :param dataset_key: key in datasets table.
    :param session: current sqlalchemy session.
    :return: a list of Image class for all of the images.

    """

    cwd = os.getcwd()
    data_dir = os.path.join(cwd, 'sql_data', "MNIST")
    mode = ['train', 'test']

    all_images = []
    for mode in mode:
        if mode.startswith('.') or not os.path.isdir(data_dir):
            continue
        df_label = pd.read_feather(data_dir+'/labels_'+mode+'.feather')
        image_dir = os.path.join(data_dir, mode)
        for image in os.listdir(image_dir):
            if image.startswith('.'):
                continue

            label = df_label.loc[df_label['id'] == image].label.values[0]
            label_map_results = session.query(LabelMap).filter(LabelMap.label.like('%' + label + '%')).first()
            if label_map_results is None:
                node_key = 9999
            else:
                node_key = label_map_results.node_key

            img = Image(dataset_key=dataset_key,
                        node_key=node_key,
                        mode=mode,
                        location=os.path.join(image_dir, image))
            all_images.append(img)
    return all_images


def get_map_dict(dataset_name):
    """return: a dictionary with key as label and value as wordnet id"""

    if dataset_name == 'MNIST':
        map_dic = {'0': '13742358',
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
        # TODO: this dictionary should be completed.
        map_dic = {'streetcar': '104342573', 'castle': '102983900', 'bicycle': '102837983', 'motorcycle': '103796045'}
    return map_dic


def add_dataset(dataset_info):
    """insert the dataset and its dependencies such as label map and directories of its images to corresponding tables.

    :param dataset_info: a dictionary with two keys: 'dataset_name' and 'nb_classes', which represents number of classes.
    :return:
    """
    Base.metadata.create_all(engine)
    session = Session()

    dataset_name = dataset_info['dataset_name']
    dataset_nb_classes = dataset_info['nb_classes']

    ########## Insert dataset info into datasets table #########
    dataset_key = session.query(Dataset.key).filter_by(name=dataset_name).first()

    if dataset_key is not None:
        dataset_key = dataset_key[0]
        dataset = session.query(Dataset).filter_by(key=dataset_key).first()
        print('dataset already exits!')
    else:
        dataset = Dataset(name= dataset_name, nb_classes=dataset_nb_classes)
        session.add(dataset)
        session.commit()
        print(dataset_name, 'added to datasets tables.')

    ########## Fill out label_maps dataset with information for dataset #########
    map_dict = get_map_dict(dataset_name)

    all_label_maps = get_label_map(map_dict, dataset_key, session)

    for label_map in all_label_maps:
        label_map.dataset = dataset
        if label_map.node_key != 9999:
            label_map.node = session.query(Node).filter(Node.key == label_map.node_key).first()

    session.add_all(all_label_maps)
    session.commit()
    print("label maps related to", dataset_name, "added to label_maps dataset.")

    # ########## fill out images dataset #########
    all_images = get_image(dataset_key, dataset_name, session)

    for image in all_images:
        image.dataset = dataset
        if image.node_key != 9999:
            image.node = session.query(Node).filter(Node.key == image.node_key).first()
    session.add_all(all_images)
    session.commit()
    print("all images of", dataset_name, "are added to images dataset.")


def main():

    ########## Generate database schema #########
    MNIST_info = {'dataset_name': 'MNIST',
                  'nb_classes': 10}
    add_dataset(MNIST_info)


    CIFAR100_info = {'dataset_name': 'CIFAR100',
                     'nb_classes': 100}
    add_dataset(CIFAR100_info)


if __name__ == "__main__":
    main()
