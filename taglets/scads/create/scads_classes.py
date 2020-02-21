import sqlalchemy as sa
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, Integer, String, ForeignKey, Boolean
from sqlalchemy.orm import relationship

Base = declarative_base()
engine = sa.create_engine('sqlite:///../../sql_data/scads_db.db')
Session = sessionmaker(bind=engine)


class Edge(Base):
    """
    An edge between two nodes in SCADS.
    See https://github.com/commonsense/conceptnet5/wiki/Edges

    key: The unique key of edge in 'edges' table
    relation_key: The key of the relation expressed by the edge. A foreign key from 'relations' table
    start_node_key: The node at the start of the edge. A foreign key from 'nodes' table
    end_node_key: The node at the end of the edge. A foreign key from 'nodes' table
    weight: The weight of the edge
    info: Additional information about the edge
    """
    __tablename__ = 'edges'
    id = Column(Integer, primary_key=True)
    relation_type = Column(Integer, ForeignKey('relations.id'))
    start_node = Column(Integer, ForeignKey('nodes.id'))
    end_node = Column(Integer, ForeignKey('nodes.id'))

    relation = relationship("Relation")

    def __repr__(self):
        return "<Edge(key='%s', relation='%s',start_node='%s', end_node='%s')>" % \
               (self.id, self.relation_type, self.start_node, self.end_node)


class Node(Base):
    """
    A node in SCADS.

    For example:
        node = Node(key='0', name='/c/en/...')
    key: The unique key of node in 'nodes' table
    name: The name of the node
    """
    __tablename__ = 'nodes'
    id = Column(Integer, primary_key=True)
    conceptnet_id = Column(String)

    images = relationship("Image", back_populates="node")
    outgoing_edges = relationship("Edge", primaryjoin=id == Edge.end_node)

    def __repr__(self):
        return "<Node(key='%s', name='%s'')>" % (self.id, self.conceptnet_id)


class Relation(Base):
    """
    A relation between two nodes in SCADS.
    See https://github.com/commonsense/conceptnet5/wiki/Relations.

    key: The unique key of relation type in 'relations' table
    name: The name of the relation
    type: The type of relation between two nodes. For example 'HasA', 'IsA', and 'Antonym'.
    """
    __tablename__ = 'relations'
    id = Column(Integer, primary_key=True)
    type = Column(String)
    is_directed = Column(Boolean)

    edges = relationship("Edge", back_populates="relation")

    def __repr__(self):
        return "<Relation(key='%s', type='%s', is_directed='%s')>" % (self.id, self.type, self.is_directed)


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
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String)
    path = Column(String)
    
    images = relationship("Image", back_populates="dataset")
    
    def __repr__(self):
        return "<Dataset(name='%s')>" % self.name


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
    id = Column(Integer, primary_key=True, autoincrement=True)
    dataset_id = Column(Integer, ForeignKey('datasets.id'))
    node_id = Column(Integer, ForeignKey('nodes.id'))
    path = Column(String)
    
    dataset = relationship("Dataset", back_populates="images")
    node = relationship("Node", back_populates="images")

    def __repr__(self):
        return "<Image(dataset='%s', node='%s', is_training = '%s', path='%s'')>" % (self.dataset_id,
                                                                                     self.node_id,
                                                                                     self.is_training,
                                                                                     self.path)
