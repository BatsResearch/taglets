import sqlalchemy as sa
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from ..create import Node
from .scads_node import ScadsNode

Base = declarative_base()
engine = sa.create_engine('sqlite:///sql_data/scads_db.db')
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
    def get_node(concept):
        """
        Get a ScadsNode given a concept.
        :return: The ScadsNode
        """
        if Scads.session is None:
            raise RuntimeError("Session is not opened.")
        sql_node = Scads.session.query(Node).filter(Node.name == concept).first()
        return ScadsNode(sql_node, Scads.session)
