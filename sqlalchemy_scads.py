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
    id = Column(Integer, primary_key=True)
    key = Column(Integer)
    name = Column(String)

    def __repr__(self):
        return "<Node(key='%d', name='%s'')>" % (self.key, self.name)


class Relation(Base):
    __tablename__ = 'relations'
    id = Column(Integer, primary_key=True)
    key = Column(Integer)
    name = Column(String)
    type = Column(String)

    edges = relationship("Edge", back_populates="relation")

    def __repr__(self):
        return "<Node(key='%d', name='%s', type='%s')>" % (self.key, self.name, self.type)


class Edge(Base):
    __tablename__ = 'edges'
    id = Column(Integer, primary_key=True)
    key = Column(Integer)
    URI = Column(String)
    relation_id = Column(Integer, ForeignKey('relations.id'))
    start_node_id = Column(Integer, ForeignKey('nodes.id'))
    end_node_id = Column(Integer, ForeignKey('nodes.id'))
    weight = Column(sa.FLOAT)
    info = Column(String)
    relation = relationship("Relation", back_populates="edges")

    def __repr__(self):
        return "<Edge(key='%d', URI='%s', relation_id='%d',start_node_id='%d', end_node_id='%d'," \
               "weight='%f', info %s ')>" % (self.key, self.URI, self.relation_id, self.start_node_id,
                                             self.end_node_id, self.weight, self.info)


def get_relations():
    cwd = os.getcwd()
    all_relations = []
    with open(Path(cwd+'/sql_data/relations.csv')) as relation_file:
        for line in relation_file:
            if line.strip():
                r_key, r_name, r_type = line.strip().split()
                relation = Relation(key=r_key, name=r_name, type=r_type)
                all_relations.append(relation)
    return all_relations


def get_nodes():
    cwd = os.getcwd()
    all_nodes = []
    with open(Path(cwd+'/sql_data/nodes.csv')) as node_file:
        for line in node_file:
            if line.strip():
                n_key, n_name = line.strip().split()
                relation = Node(key=n_key, name=n_name)
                all_nodes.append(relation)
    return all_nodes


def get_edges():
    cwd = os.getcwd()
    all_edges = []
    with open(Path(cwd+'/sql_data/edges.csv')) as edge_file:
        for line in edge_file:
            if line.strip():
                key, URI, relation_id, start_node_id, end_node_id,weight = line.strip().split()[:6]
                info = " " .join(line.strip().split()[6:])
                relation = Edge(key=key,URI=str(URI),relation_id=relation_id, start_node_id=start_node_id,
                                end_node_id=end_node_id, weight=weight, info = info)
                all_edges.append(relation)
    return all_edges


def main():

    # generate database schema
    Base.metadata.create_all(engine)

    session = Session()

    all_relations = get_relations()
    session.add_all(all_relations)

    all_nodes = get_nodes()
    session.add_all(all_nodes)

    all_edges = get_edges()
    session.add_all(all_edges)

    session.commit()
    session.close()


if __name__ == "__main__":
    main()
