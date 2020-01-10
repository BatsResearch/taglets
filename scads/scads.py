import sqlalchemy as sa
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy_scads import Node

Base = declarative_base()
engine = sa.create_engine('sqlite:///sql_data/scads_db.db')
Session = sessionmaker(bind=engine)


class Scads:
    """
    Class to interact with the database
    Structured Collections of Annotated Data Sets (SCADS)
    """

    @staticmethod
    def get_node(concept):
        """
        Jeff: It's possible that there are different wornet_ids corresponding to a single concept.
        In that case, it will be difficult to specify which node the user wants.
        Get a ScadsNode given a concept
        :return: The ScadsNode
        """
        session = Session()
        node = session.query(Node).filter(Node.name == concept).first()
        session.close()
        return node
