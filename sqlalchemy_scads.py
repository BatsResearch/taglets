import numpy as np
import sqlalchemy as sa
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Table, Column, Integer, String, MetaData, ForeignKey
from sqlalchemy.orm import relationship, backref
import os
from pathlib import Path


Base = declarative_base()
engine = sa.create_engine('sqlite:///sql_data/scads_db.db')
Session = sessionmaker(bind=engine)


class Node(Base):
    __tablename__ = 'nodes'
    key = Column(Integer, primary_key=True)
    name = Column(String)
    
    in_edges = relationship("Edge", back_populates="end_node")

    def __repr__(self):
        return "<Node(key='%s', name='%s'')>" % (self.key, self.name)

class Relation(Base):
    __tablename__ = 'relations'
    key = Column(Integer, primary_key=True)
    name = Column(String)
    type = Column(String)

    edges = relationship("Edge", back_populates="relation")

    def __repr__(self):
        return "<Relation(key='%s', name='%s', type='%s')>" % (self.key, self.name, self.type)

class Edge(Base):
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
    __tablename__ = 'datasets'
    key = Column(Integer, primary_key=True,autoincrement=True)
    name = Column(String)
    nb_classes = Column(Integer)
    def __repr__(self):
        return "<Dataset(name='%s', nb_classes='%s')>" % (self.name, self.nb_classes)

class Image(Base):
    __tablename__ = 'images'
    key = Column(Integer, primary_key=True, autoincrement=True)
    dataset_key =  Column(Integer, ForeignKey('relations.key'))
    node_key = Column(Integer, ForeignKey('nodes.key'))
    mode = Column(String) # train, test, None
    location = Column(String)

    def __repr__(self):
        return "<Image(dataset_key='%s', node_key='%s',mode = '%s', location='%s'')>" % ( self.dataset_key,self.node_key,
                                                                                       self.mode, self.location)

class Label_map(Base):
    __tablename__ = 'label_maps'
    key=Column(Integer, primary_key=True, autoincrement=True)
    label=Column(String)
    node_key=Column(Integer, ForeignKey('nodes.key'))
    dataset_key=Column(Integer, ForeignKey('datasets.key'))
    def __repr__(self):
        return "<Image(label = '%s', node_key='%s', dataset_key='%s')>" % (self.label, self.node_key,
                                                                                    self.dataset_key)

def get_relations():
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
    cwd = os.getcwd()
    all_nodes = []
    with open(os.path.join(cwd, 'sql_data', 'nodes.csv'), encoding="utf8") as node_file:
        for line in node_file:
            if line.strip():
                n_key, n_name = line.strip().split()
                relation = Node(key=n_key, name=n_name)
                all_nodes.append(relation)
    return all_nodes

def get_edges():
    cwd = os.getcwd()
    all_edges = []
    with open(os.path.join(cwd, 'sql_data', 'edges.csv'), encoding="utf8") as edge_file:
        for line in edge_file:
            if line.strip():
                key, URI, relation_key, start_node_key, end_node_key,weight = line.strip().split()[:6]
                info = " " .join(line.strip().split()[6:])
                relation = Edge(key=key,URI=str(URI),relation_key=relation_key, start_node_key=start_node_key,
                                end_node_key=end_node_key, weight=weight, info=info)
                all_edges.append(relation)
    return all_edges

def get_label_map(map_dict,dataset_key,session):
    '''
    :param map_dict: key is label and value is wordnet id. e.g., {'streetcar':104342573,  'castle': 102983900}
    :param dataset_key: key of the dataset, e.g., key of cifar100 in datasets table is 0
    :param session:
    :return: a list of 'Label_map' object that should be inserted into 'label_maps' dataset
    '''
    all_labels = []
    for label_name,wordnet_id in map_dict.items():
        node_results = session.query(Node).filter(Node.name.like('%'+wordnet_id+'%'))
        node_key = 99999
        for r in node_results:
            node_key = r.key
        label_map = Label_map(label=label_name, node_key=node_key, dataset_key = dataset_key )
        all_labels.append(label_map)
    return all_labels

def get_cifar100(cifar_key,session):
    '''
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
    mode is train or test or none (image in dataset is in test folder or train folder; otherwise it should be none).
    :param cifar_key: key of cifar dataset
    :param session:
    :return: list of directories of all cifar images
    '''
    cwd = os.getcwd()
    cifar100_dir = os.path.join(cwd, 'sql_data','CIFAR100')
    all_images = []
    for mode in os.listdir(cifar100_dir):
        if mode.startswith('.'):
            continue
        class_dir = os.path.join(cifar100_dir,mode)
        for label in os.listdir(class_dir):
            if label.startswith('.'):
                continue
            label_map_results = session.query(Label_map).filter(Label_map.label.like('%' + label + '%')).first()
            if label_map_results == None:
                node_key = 9999
            else:
                node_key = label_map_results.node_key
            image_dir = os.path.join(class_dir,label)
            for image in os.listdir(image_dir):
                img = Image(dataset_key=cifar_key, node_key=node_key, mode = mode, location=os.path.join(image_dir,image))
                all_images.append(img)

    return all_images

def get_map_dic_cifar():
    '''
    :return: a dictionary with key as label and value as wordnet id
    '''
    map_dic = {'streetcar': '104342573', 'castle': '102983900','bicycle':'102837983','motorcycle':'103796045'}
    return map_dic

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
        all_nodes[int(edge.end_node_key)].in_edges.append(edge)


    session.add_all(all_relations)
    session.add_all(all_nodes)
    session.add_all(all_edges)
    session.commit()


    ########## insert cifar100 info into datasets table #########
    cifar100 =  Dataset(key=0, name='CIFAR100',nb_classes=100)
    session.add(cifar100)
    session.commit()

    ########## fill label_maps dataset with information from CIFAR100 #########
    cifar_id = session.query(Dataset.key).filter_by(name='CIFAR100').first()
    cifar_key = cifar_id._asdict()['key']
    cifar_map_dict = get_map_dic_cifar()
    all_label_maps = get_label_map(cifar_map_dict,cifar_key,session)
    session.add_all(all_label_maps)
    session.commit()

    ########## fill out images dataset with CIFAR100 images #########
    all_images = get_cifar100(cifar_key,session)
    session.add_all(all_images)
    session.commit()

    session.commit()
    session.close()


if __name__ == "__main__":
    main()
