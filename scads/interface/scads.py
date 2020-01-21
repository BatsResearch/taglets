import sqlalchemy as sa
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from scads.build.scads_classes import Node
from scads.interface.scads_node import ScadsNode

Base = declarative_base()
engine = sa.create_engine('sqlite:///sql_data/scads_db.db')
Session = sessionmaker(bind=engine)


class Scads:
    """
    Class to interact with the database
    Structured Collections of Annotated Data Sets (SCADS)
    """

    session = None

    @staticmethod
    def open():
        Scads.session = Session()

    @staticmethod
    def close():
        Scads.session.close()

    @staticmethod
    def get_node(concept):
        """
        Jeff: It's possible that there are different wornet_ids corresponding to a single concept.
        In that case, it will be difficult to specify which node the user wants.
        Get a ScadsNode given a concept
        :return: The ScadsNode
        """
        if Scads.session is None:
            raise RuntimeError("Session is not opened.")
        sql_node = Scads.session.query(Node).filter(Node.name == concept).first()
        return ScadsNode(sql_node, Scads.session)
