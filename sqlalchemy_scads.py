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


def main():

    # generate database schema
    Base.metadata.create_all(engine)

    session = Session()

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
    session.close()


if __name__ == "__main__":
    main()
