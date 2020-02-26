import sqlalchemy as sa
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from ..create import Node
from .scads_node import ScadsNode

PATH_TO_DATABASE = './test_data/test_scads_db.db'
Base = declarative_base()
engine = sa.create_engine('sqlite:///' + PATH_TO_DATABASE)
Session = sessionmaker(bind=engine)


class Scads:
    """
    A class providing connection to Structured Collections of Annotated Data Sets (SCADS).
    """
    session = None
    label_to_concept = {}   # TODO: Complete this

    @staticmethod
    def open():
        Scads.session = Session()

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
        sql_node = Scads.session.query(Node).filter(Node.conceptnet_id == conceptnet_id).first()
        if not sql_node:
            raise Exception("Node not found.")
        return ScadsNode(sql_node, Scads.session)
