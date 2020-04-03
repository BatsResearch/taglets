import unittest
import os
from taglets.scads import Scads
from taglets.scads.create.install import Installer, CifarInstallation, MnistInstallation

DB_PATH = "test/scads/test_data/test_scads.db"
CONCEPTNET_PATH = "test/scads/test_data"
CIFAR_PATH = "test/scads/test_data/CIFAR100"
MNIST_PATH = "test/scads/test_data/MNIST"


class TestSCADS(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        installer = Installer(DB_PATH)
        installer.install_conceptnet(CONCEPTNET_PATH)
        installer.install_dataset(CIFAR_PATH, CifarInstallation())
        installer.install_dataset(MNIST_PATH, MnistInstallation())
        Scads.open(DB_PATH)

    def test_invalid_conceptnet_id(self):
        with self.assertRaises(Exception):
            Scads.get_node_by_conceptnet_id("invalid")

    def test_filters_non_english(self):
        with self.assertRaises(Exception):
            Scads.get_node_by_conceptnet_id("/c/test/test")

    def test_valid_node1(self):
        node = Scads.get_node_by_conceptnet_id("/c/en/zero/n/wn/quantity")
        self.assertEqual(node.node.id, 0)
        self.assertEqual(node.get_conceptnet_id(), "/c/en/zero/n/wn/quantity")

        self.assertEqual(node.get_datasets(), ['MNIST'])

        images = node.get_images()
        self.assertEqual(len(images), 3)
        self.assertTrue('test/scads/test_data/MNIST/test/scads/test_data/MNIST/train/21.png'
                        in images)

        neighbors = node.get_neighbors()
        self.assertEqual(len(neighbors), 7)
        self.assertEqual(neighbors[0].get_end_node().node.id, 1)

    def test_valid_node2(self):
        node = Scads.get_node_by_conceptnet_id("/c/en/apple")
        self.assertEqual(node.node.id, 10)
        self.assertEqual(node.get_conceptnet_id(), "/c/en/apple")

        self.assertEqual(node.get_datasets(), ['CIFAR100'])

        images = node.get_images()
        self.assertEqual(len(images), 15)
        self.assertTrue('test/scads/test_data/CIFAR100/test/scads/test_data/CIFAR100/test/apple/0001 copy.png'
                        in images)

        neighbors = node.get_neighbors()
        self.assertEqual(len(neighbors), 1)
        self.assertEqual(neighbors[0].get_end_node().node.id, 11)

    def test_undirected_relation(self):
        node = Scads.get_node_by_conceptnet_id("/c/en/zero/n/wn/quantity")
        relation = node.get_neighbors()[0]

        self.assertEqual(relation.get_relationship(), "/r/Antonym")
        self.assertFalse(relation.is_directed())
        self.assertIsNotNone(relation.get_end_node())

    def test_directed_relation(self):
        node = Scads.get_node_by_conceptnet_id("/c/en/zero/n/wn/quantity")
        relation = node.get_neighbors()[1]

        self.assertEqual(relation.get_relationship(), "/r/AtLocation")
        self.assertTrue(relation.is_directed())
        self.assertIsNotNone(relation.get_end_node())

    def test_neighbors(self):
        node = Scads.get_node_by_conceptnet_id("/c/en/zero/n/wn/quantity")
        neighbor1 = node.get_neighbors()[0].get_end_node()
        self.assertEqual(neighbor1.node.id, 1)
        self.assertEqual(node.node.id, neighbor1.get_neighbors()[0].get_end_node().node.id)
        self.assertEqual(len(neighbor1.get_images()), 3)

        neighbor2 = node.get_neighbors()[1].get_end_node()
        self.assertEqual(neighbor2.node.id, 2)
        self.assertEqual(neighbor2.get_datasets(), [])
        self.assertEqual(len(neighbor2.get_images()), 0)

    @classmethod
    def tearDownClass(cls):
        Scads.close()
        os.remove(DB_PATH)


if __name__ == "__main__":
    unittest.main()
