import sqlalchemy as sa
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship

Base = declarative_base()
engine = sa.create_engine('sqlite:///sql_data/scads_db.db')
Session = sessionmaker(bind=engine)


class Node(Base):
    """
    A node in SCADS.

    For example:
        node = Node(key='0', name='apple')
    key: The unique key of node in 'nodes' table
    name: The name of the node
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
    """
    A relation between two nodes in SCADS.
    See https://github.com/commonsense/conceptnet5/wiki/Relations.

    key: The unique key of relation type in 'relations' table
    name: The name of the relation
    type: The type of relation between two nodes. For example 'HasA', 'IsA', and 'Antonym'.
    """
    __tablename__ = 'relations'
    key = Column(Integer, primary_key=True)
    name = Column(String)
    type = Column(String)
    edges = relationship("Edge", back_populates="relation")

    def __repr__(self):
        return "<Relation(key='%s', name='%s', type='%s')>" % (self.key, self.name, self.type)


class Edge(Base):
    """
    An edge between two nodes in SCADS.
    See https://github.com/commonsense/conceptnet5/wiki/Edges

    key: The unique key of edge in 'edges' table
    URI: The URI of the whole edge. See https://github.com/commonsense/conceptnet5/wiki/URI-hierarchy
    relation_key: The key of the relation expressed by the edge. A foreign key from 'relations' table
    start_node_key: The node at the start of the edge. A foreign key from 'nodes' table
    end_node_key: The node at the end of the edge. A foreign key from 'nodes' table
    weight: The weight of the edge
    info: Additional information about the edge
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
    """
    A dataset in SCADS.

    We could have several datasets such as ImageNet, CIFAR10, CIFAR100, MNIST, etc. in SCADS.
    For example:
        cifar100 = Dataset(key=0, name='CIFAR100', nb_classes=100)
    key: The unique, autoincremented key of the dataset
    name: The name of the dataset
    nb_classes: The total number of classes in the dataset
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
    """
    An Image in SCADS.

    For example:
          apple_img_1 = Image(dataset_key='cifar100 key',
                              node_key='apple key',
                              mode = 'train',
                              location='directory/apple_1.png')

    key: The unique, autoincremented key of the image
    dataset_key: The key of the corresponding dataset for the image. A foreign key from 'datasets' table
    node_key: The key of the corresponding node for the image. A foreign key from 'nodes' table
    mode: If the image is saved for 'train' or 'test' in actual dataset
          If None, the image is not categorized based on train and test
    location: The location of the image
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
    """
    A mapping between dataset class names and nodes.

    Each class name in dataset has a corresponding node in nodes table. This class represents the association between
    a class name in each dataset with its corresponding node.

    For example:
        map_apple = LabelMap(label='apple', node_key='apple_node_key', dataset_key = 'cifar100_key')
    Associates the label name 'apple' in CIFAR100 with the corresponding 'apple' in nodes dataset

    key: The unique, autoincremented key of the label map
    label: The name of class label
    node_key: The key of corresponding node in 'nodes' table
    dataset_key: The key of corresponding dataset
    """
    __tablename__ = 'label_maps'
    key = Column(Integer, primary_key=True, autoincrement=True)
    label = Column(String)
    node_key = Column(Integer, ForeignKey('nodes.key'))
    dataset_key = Column(Integer, ForeignKey('datasets.key'))

    dataset = relationship("Dataset", back_populates="label_maps")
    node = relationship("Node", back_populates="label_maps")
    
    def __repr__(self):
        return "<Image(label = '%s', node_key='%s', dataset_key='%s')>" % (self.label,
                                                                           self.node_key,
                                                                           self.dataset_key)
