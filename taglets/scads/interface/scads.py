import sqlalchemy as sa
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from os import path

from ..create import Node
from .scads_node import ScadsNode

Base = declarative_base()


class Scads:
    """
    A class providing connection to Structured Collections of Annotated Data Sets (SCADS).
    """
    session = None
    #root_path = './'
    root_path = '/lwll/external/'

    @staticmethod
    def set_root_path(root_path):
        Scads.root_path = root_path

    @staticmethod
    def get_root_path():
        return Scads.root_path

    @staticmethod
    def open(path_to_database):
        if path.exists(path_to_database):
            engine = sa.create_engine('sqlite:///' + path_to_database)
            Session = sessionmaker(bind=engine)
            Scads.session = Session()
        else:
            raise RuntimeError("Invalid path to database.")

    @staticmethod
    def close():
        Scads.session.close()

    @staticmethod
    def get_node_by_conceptnet_id(conceptnet_id):
        """
        Get a ScadsNode given a concept.
        :return: The ScadsNode
        """
        if Scads.session is None:
            raise RuntimeError("Session is not opened.")
        try:
            sql_node = Scads.session.query(Node).filter(Node.conceptnet_id == conceptnet_id).first()
        except:
            raise RuntimeError("Invalid database.")
        if not sql_node:
            raise Exception("Node not found.")
        return ScadsNode(sql_node, Scads.session)
