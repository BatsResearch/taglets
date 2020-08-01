import os
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker
from taglets.scads.create.scads_classes import Node, Edge, Relation, Base
import unittest

TEST_DATA = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_data")
DB_PATH = os.path.join(TEST_DATA, "test_scads.db")


class TestModule(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Sets up test SCADS

        # Generates database schema
        engine = sa.create_engine('sqlite:///' + DB_PATH)
        Session = sessionmaker(bind=engine)
        session = Session()
        session.execute('PRAGMA foreign_keys=ON')
        Base.metadata.create_all(engine)

        # Creates relations
        rel1 = Relation(id=0, type="/r/IsA", is_directed=True)
        rel2 = Relation(id=1, type="/r/RelatedTo", is_directed=False)
        session.add(rel1)
        session.add(rel2)

        # Creates nodes

    def test_module(self):

        raise NotImplementedError
