import os
from .scads_classes import Node, Dataset
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker


def add_dataset(path_to_database, root, path_to_dataset, dataset_installer):
    """
    Insert the dataset and its dependencies (LabelMaps, Images) into the database.
    :param dataset_info: A dictionary the dataset name and the number of classes.
    :return: None
    """
    if not os.path.isdir(os.path.join(root, path_to_dataset)):
        print("Path does not exist.")

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
        print(dataset_name, 'added to datasets table.')
    else:
        print('Dataset already exits at', dataset.path)
        return

    # Add Images to database
    all_images = dataset_installer.get_images(dataset, session, root)
    for image in all_images:
        image.dataset = dataset
        if image.node_id:
            image.node = session.query(Node).filter(Node.id == image.node_id).one()
    session.add_all(all_images)
    session.commit()
    print("Images in", dataset_name, "added to images dataset.")
