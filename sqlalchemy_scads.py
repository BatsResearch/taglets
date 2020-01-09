import sqlalchemy as sa
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship
import os

Base = declarative_base()
engine = sa.create_engine('sqlite:///sql_data/scads_db.db')
Session = sessionmaker(bind=engine)


class Node(Base):
    """Represents a node in SCADS (Structured Collections of Annotated Data Sets).

    E.g.::

        node = Node(key='0', name='apple')

    :param key: the key of node in 'nodes' table. The key should be unique.
    :param name: the name of node.

    """

    __tablename__ = 'nodes'
    key = Column(Integer, primary_key=True)
    name = Column(String)
    
    in_edges = relationship("Edge", back_populates="end_node")
    images = relationship("Image", back_populates="node")
    label_maps = relationship("LabelMap", back_populates="node")

    def __repr__(self):
        return "<Node(key='%s', name='%s'')>" % (self.key, self.name)


class Relation(Base):
    """Represents types of relations between two nodes in SCADS (Structured Collections of Annotated Data Sets).

    See https://github.com/commonsense/conceptnet5/wiki/Relations

    :param key: the key of relation type in 'relations' table. The key should be unique.
    :param name: the name of relation.
    :param type: type of relation between two nodes. For example 'HasA', 'IsA', and 'Antonym' are some types of relationship.

    """

    __tablename__ = 'relations'
    key = Column(Integer, primary_key=True)
    name = Column(String)
    type = Column(String)
    edges = relationship("Edge", back_populates="relation")

    def __repr__(self):
        return "<Relation(key='%s', name='%s', type='%s')>" % (self.key, self.name, self.type)


class Edge(Base):
    """Represents edges between two nodes in SCADS (Structured Collections of Annotated Data Sets).

    See https://github.com/commonsense/conceptnet5/wiki/Edges

    :param key: the key of edge in 'edges' table. The key should be unique.
    :param URI: the URI of the whole edge. See https://github.com/commonsense/conceptnet5/wiki/URI-hierarchy
    :param relation_key: the key of the relation expressed by the edge. It is a foreign key from 'relations' table.
    :param start_node_key: the node at the start of the edge. It is a foreign key from 'nodes' table.
    :param end_node_key: the node at the end of the edge. It is a foreign key from 'nodes' table.
    :param weight: weight of the edge.
    :param info: additional information about the edge.

    """

    __tablename__ = 'edges'
    key = Column(Integer, primary_key=True)
    URI = Column(String)
    relation_key = Column(Integer, ForeignKey('relations.key'))
    start_node_key = Column(Integer)
    end_node_key = Column(Integer, ForeignKey('nodes.key'))
    weight = Column(sa.FLOAT)
    info = Column(String)
    
    relation = relationship("Relation", back_populates="edges")
    end_node = relationship("Node", back_populates="in_edges")

    def __repr__(self):
        return "<Edge(key='%s', URI='%s', relation_key='%s',start_node_key='%s', end_node_key='%s'," \
               "weight='%s', info %s ')>" % (self.key, self.URI, self.relation_key, self.start_node_key,
                                             self.end_node_key, self.weight, self.info)


class Dataset(Base):
    """ Represents datasets in SCADS (Structured Collections of Annotated Data Sets).

    We could have several datasets such as ImageNet, CIFAR10, CIFAR100, MNIST, etc. in SCADS.

    E.g.::
        cifar100 = Dataset(key=0, name='CIFAR100', nb_classes=100)

    :param key: the key of dataset in 'datasets' table. The key should be unique. Key is autoincrement.
    :param name: the name of dataset.
    :param nb_classes: total number of classes in dataset.

    """

    __tablename__ = 'datasets'
    key = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String)
    nb_classes = Column(Integer)
    
    images = relationship("Image", back_populates="dataset")
    label_maps = relationship("LabelMap", back_populates="dataset")
    
    def __repr__(self):
        return "<Dataset(name='%s', nb_classes='%s')>" % (self.name, self.nb_classes)


class Image(Base):
    r""" Represents the single image information in SCADS (Structured Collections of Annotated Data Sets).

    E.g.::

          apple_img_1 = Image(dataset_key='cifar100 key', node_key='apple key', mode = 'train', location='directory/apple_1.png')

      Indicates that the an image of apple (with the corresponding apple key in nodes dataset) for cifar100 dataset is
      located in directory/apple_1.png

    :param key: the key of image in 'images' table. The key should be unique. Key is autoincrement.
    :param dataset_key: the key of the corresponding dataset for the image. It is a foreign key from 'datasets' table.
    :param node_key: the key of the corresponding node for the image. It is a foreign key from 'nodes' table.
    :param mode: shows if the image is saved for 'train' or 'test' in actual dataset. 'None' means in dataset,
    the image is not categorized based on train and test.
    :param location: the directory in which the image is located.

    """

    __tablename__ = 'images'
    key = Column(Integer, primary_key=True, autoincrement=True)
    dataset_key = Column(Integer, ForeignKey('datasets.key'))
    node_key = Column(Integer, ForeignKey('nodes.key'))
    mode = Column(String)   # train, test, None
    location = Column(String)
    
    dataset = relationship("Dataset", back_populates="images")
    node = relationship("Node", back_populates="images")

    def __repr__(self):
        return "<Image(dataset_key='%s', node_key='%s',mode = '%s', location='%s'')>" % (self.dataset_key,
                                                                                         self.node_key,
                                                                                         self.mode,
                                                                                         self.location)


class LabelMap(Base):
    """Represents the mapping between the class names in datasets and nodes.

    Each class name in dataset has a corresponding node in nodes table. This class shows the association between class
    name in each dataset with node in nodes dataset.

    E.g.::
        map_apple = LabelMap(label='apple', node_key='apple_node_key', dataset_key = 'cifar100_key')

    Associates the label name 'apple' in cifar100 dataset with the correspoinding 'apple' in nodes dataset.

    :param key: the key of map in 'label_maps' table. The key should be unique. Key is autoincrement.
    :param label: name of class label.
    :param node_key: the key of corresponding node in 'nodes' table.
    :param dataset_key: the key of corresponding dataset.

    """

    __tablename__ = 'label_maps'
    key = Column(Integer, primary_key=True, autoincrement=True)
    label = Column(String)
    node_key = Column(Integer, ForeignKey('nodes.key'))
    dataset_key = Column(Integer, ForeignKey('datasets.key'))

    dataset = relationship("Dataset", back_populates="label_maps")
    node = relationship("Node", back_populates="label_maps")
    
    def __repr__(self):
        return "<Image(label = '%s', node_key='%s', dataset_key='%s')>" % (self.label, self.node_key,
                                                                                    self.dataset_key)


def get_relations():
    """ load the csv file containing all relations in conceptnet, and return a list of Relation class."""

    cwd = os.getcwd()
    all_relations = []
    with open(os.path.join(cwd, 'sql_data', 'relations.csv'), encoding="utf8") as relation_file:
        for line in relation_file:
            if line.strip():
                r_key, r_name, r_type = line.strip().split()
                relation = Relation(key=r_key, name=r_name, type=r_type)
                all_relations.append(relation)
    return all_relations


def get_nodes():
    """ load the csv file containing all nodes in conceptnet, and return a list of 'Node' class"""

    cwd = os.getcwd()
    all_nodes = []
    with open(os.path.join(cwd, 'sql_data', 'nodes.csv'), encoding="utf8") as node_file:
        for line in node_file:
            if line.strip():
                n_key, n_name = line.strip().split()
                node = Node(key=n_key, name=n_name)
                all_nodes.append(node)
    return all_nodes


def get_edges():
    """ load the csv file containing all edges in conceptnet, and return a list of 'Edge' class."""

    cwd = os.getcwd()
    all_edges = []
    with open(os.path.join(cwd, 'sql_data', 'edges.csv'), encoding="utf8") as edge_file:
        for line in edge_file:
            if line.strip():
                key, URI, relation_key, start_node_key, end_node_key,weight = line.strip().split()[:6]
                info = " " .join(line.strip().split()[6:])
                edge = Edge(key=key,URI=str(URI),relation_key=relation_key, start_node_key=start_node_key,
                                end_node_key=end_node_key, weight=weight, info=info)
                all_edges.append(edge)
    return all_edges


def main():

    ########## generate database schema #########
    Base.metadata.create_all(engine)

    session = Session()

    ########## read nodes, edges, and relations from files and insert into corresponding tables #########
    all_relations = get_relations()
    all_nodes = get_nodes()
    all_edges = get_edges()

    for edge in all_edges:
        print(edge.end_node_key)

    # Top: this assumes the relations in all_relations are already SORTED based on their keys and the keys start at 0
    for edge in all_edges:
        edge.relation = all_relations[int(edge.relation_key)]
        edge.end_node = all_nodes[int(edge.end_node_key)]

    session.add_all(all_relations)
    session.add_all(all_nodes)
    session.add_all(all_edges)
    session.commit()
    session.close()


if __name__ == "__main__":
    main()
