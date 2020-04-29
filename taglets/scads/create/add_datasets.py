import os
from .scads_classes import Node, Dataset
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker
import logging


def add_dataset(path_to_database, root, path_to_dataset, dataset_installer):
    """
    Insert the dataset and its dependencies (LabelMaps, Images) into the database.
    :param dataset_info: A dictionary the dataset name and the number of classes.
    :return: None
    """
    logging.getLogger().setLevel(logging.INFO)
    if not os.path.isdir(os.path.join(root, path_to_dataset)):
        logging.error("Path does not exist.")
        return

    # Get session
    engine = sa.create_engine('sqlite:///' + path_to_database)
    Session = sessionmaker(bind=engine)
    session = Session()

    # Add Dataset to database
    dataset_name = dataset_installer.get_name()
    dataset = session.query(Dataset).filter_by(name=dataset_name).first()
    if not dataset:
        dataset = Dataset(name=dataset_name, path=path_to_dataset)
        session.add(dataset)
        session.commit()
        logging.info(dataset_name, 'added to datasets table.')
    else:
        logging.warning('Dataset already exits at', dataset.path)
        return

    # Add Images to database
    all_images = dataset_installer.get_images(dataset, session, root)
    session.add_all(all_images)
    session.commit()
    logging.info("Images in", dataset_name, "added to images dataset.")
