import os
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker
from taglets.task import Task
from taglets.scads import Scads
from taglets.scads.create.scads_classes import Node, Edge, Relation, Dataset, Image, Base
from torchvision import transforms
from torchvision.datasets import ImageFolder

TEST_DATA = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../test_data/modules")
DB_PATH = os.path.join(TEST_DATA, "test_module.db")


class HiddenLabelDataset(Dataset):
    """
    Wraps a labeled dataset so that it appears unlabeled
    """
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        return img

    def __len__(self):
        return len(self.dataset)


class TestModule:

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
        is_a = Relation(id=0, type="/r/IsA", is_directed=True)
        related_to = Relation(id=1, type="/r/RelatedTo", is_directed=False)
        session.add_all((is_a, related_to))

        # Creates nodes
        airplane = Node(id=0, conceptnet_id="/c/en/airplane")
        cat = Node(id=1, conceptnet_id="/c/en/cat")
        dog = Node(id=2, conceptnet_id="/c/en/dog")
        lion = Node(id=3, conceptnet_id="/c/en/lion")
        propeller_plane = Node(id=4, conceptnet_id="/c/en/propeller_plane")
        wolf = Node(id=5, conceptnet_id="/c/en/wolf")
        session.add_all((airplane, cat, dog, lion, propeller_plane, wolf))

        # Creates edges
        edge0 = Edge(id=0, weight=1.0)
        edge0.relation = related_to
        edge0.start_node = airplane.id
        edge0.end_node = propeller_plane.id

        edge1 = Edge(id=1, weight=1.0)
        edge1.relation = related_to
        edge1.start_node = cat.id
        edge1.end_node = lion.id

        edge2 = Edge(id=2, weight=2.0)
        edge2.relation = related_to
        edge2.start_node = dog.id
        edge2.end_node = wolf.id

        edge3 = Edge(id=3, weight=0.5)
        edge3.relation = related_to
        edge3.start_node = lion.id
        edge3.end_node = wolf.id

        edge4 = Edge(id=4, weight=1.0)
        edge4.relation = is_a
        edge4.start_node = propeller_plane.id
        edge4.end_node = airplane.id

        session.add_all((edge0, edge1, edge2, edge3, edge4))

        # Creates dataset
        related = Dataset(id=0, name="related", path=os.path.join(TEST_DATA, "related"))
        session.add(related)

        # Creates images
        images = {
            (lion, "lion"):
                ["imagenet_n02129165_n02129165_1002.jpg",
                 "imagenet_n02129165_n02129165_10000.jpg",
                 "imagenet_n02129165_n02129165_10004.jpg",
                 "imagenet_n02129165_n02129165_10019.jpg",
                 "imagenet_n02129165_n02129165_10029.jpg"],
            (propeller_plane, "propeller_plane"):
                ["imagenet_n04012084_n04012084_10003.jpg",
                 "imagenet_n04012084_n04012084_10008.jpg",
                 "imagenet_n04012084_n04012084_10010.jpg",
                 "imagenet_n04012084_n04012084_10011.jpg",
                 "imagenet_n04012084_n04012084_10015.jpg"],
            (wolf, "wolf"):
                ["imagenet_n02114100_n02114100_10.jpg",
                 "imagenet_n02114100_n02114100_10001.jpg",
                 "imagenet_n02114100_n02114100_10011.jpg",
                 "imagenet_n02114100_n02114100_10013.jpg",
                 "imagenet_n02114100_n02114100_10019.jpg"]
        }

        for (node, name), paths in images.items():
            for path in paths:
                image = Image(path=os.path.join("related", name, path))
                image.dataset = related
                image.node = node
                session.add(image)

        # Commits data
        session.commit()

    def setUp(self):
        preprocess = transforms.Compose(
            [transforms.CenterCrop(224),
             transforms.ToTensor()])

        self.train = ImageFolder(os.path.join(TEST_DATA, "train"), transform=preprocess)
        self.val = ImageFolder(os.path.join(TEST_DATA, "val"), transform=preprocess)
        self.test = ImageFolder(os.path.join(TEST_DATA, "test"), transform=preprocess)
        self.unlabeled = ImageFolder(os.path.join(TEST_DATA, "unlabeled"), transform=preprocess)
        self.unlabeled = HiddenLabelDataset(self.unlabeled)

        self.task = Task("test_module", ["/c/en/airplane", "/c/en/cat", "/c/en/dog"],
                         (224, 224), self.train, self.unlabeled, self.val,
                         scads_path=DB_PATH)
        Scads.set_root_path(TEST_DATA)

    def test_module(self):
        module = self._get_module(self.task)
        module.train_taglets(self.train, self.val, False)
        taglets = module.get_taglets()
        for taglet in taglets:
            votes = taglet.execute(self.unlabeled, False)
            self.assertEqual(len(votes), 15)
            for vote in votes:
                self.assertTrue(vote in (0, 1, 2))
            self.assertGreater(taglet.evaluate(self.test, False), 0.5)

    def _get_module(self, task):
        """Constructs a Module with a given task for testing.

        This method lets Module-specific unit tests inherit the tests from this
        abstract parent test class.

        :param task: the Task to be used for testing
        :return: the Module to be tested
        """
        raise NotImplementedError

    @classmethod
    def tearDownClass(cls):
        os.remove(DB_PATH)
