import unittest
import os
from taglets.scads import Scads
from taglets.scads.create.install import Installer, CifarInstallation, MnistInstallation, ImageNetInstallation, COCO2014Installation, GoogleOpenImageInstallation, VOC2009Installation, DomainNetInstallation

ROOT = os.path.dirname(os.path.realpath(__file__))
ROOT = ROOT + "/../test_data/scads"
DB_PATH = ROOT + "/test_scads.db"
CONCEPTNET_PATH = ROOT + "/conceptnet"
CIFAR_PATH = "cifar100"
MNIST_PATH = "mnist"
IMAGENET_PATH = "imagenet_1k"
COCO2014_PATH = "coco2014"
GOOGLE_OPEN_IMAGE_PATH = "google_open_image"
VOC2009_PATH = "voc2009"
domainnet_PATH = "domainnet"


class TestSCADS(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        installer = Installer(DB_PATH)
        installer.install_conceptnet(CONCEPTNET_PATH)
        installer.install_dataset(ROOT, CIFAR_PATH, CifarInstallation())
        installer.install_dataset(ROOT, MNIST_PATH, MnistInstallation())
        installer.install_dataset(ROOT, IMAGENET_PATH, ImageNetInstallation())
        installer.install_dataset(ROOT, COCO2014_PATH, COCO2014Installation())
        installer.install_dataset(ROOT, GOOGLE_OPEN_IMAGE_PATH, GoogleOpenImageInstallation())
        installer.install_dataset(ROOT, VOC2009_PATH, VOC2009Installation())
        installer.install_dataset(ROOT, domainnet_PATH, DomainNetInstallation('clipart'))


        Scads.open(DB_PATH)

    def test_invalid_conceptnet_id(self):
        with self.assertRaises(Exception):
            Scads.get_node_by_conceptnet_id("invalid")

    def test_filters_non_english(self):
        with self.assertRaises(Exception):
            Scads.get_node_by_conceptnet_id("/c/test/test")

    def test_valid_node1(self):
        node = Scads.get_node_by_conceptnet_id("/c/en/one")
        self.assertEqual(node.node.id, 1)
        self.assertEqual(node.get_conceptnet_id(), "/c/en/one")

        self.assertEqual(node.get_datasets(), ['MNIST'])

        images = node.get_images()
        self.assertEqual(len(images), 2)
        self.assertTrue('mnist/mnist_full/test/9994.png'
                        in images)

        neighbors = node.get_neighbors()
        self.assertEqual(len(neighbors), 2)
        self.assertEqual(neighbors[0].get_end_node().node.id, 0)
        self.assertEqual(neighbors[1].get_end_node().node.id, 5)

    def test_valid_node2(self):
        node = Scads.get_node_by_conceptnet_id("/c/en/wardrobe")
        self.assertEqual(node.node.id, 10)
        self.assertEqual(node.get_conceptnet_id(), "/c/en/wardrobe")

        self.assertEqual(node.get_datasets(), ['CIFAR100'])

        images = node.get_images()
        self.assertEqual(len(images), 1)
        self.assertTrue('cifar100/cifar100_full/train/41904.png'
                        in images)

        neighbors = node.get_neighbors()
        self.assertEqual(len(neighbors), 1)
        self.assertEqual(neighbors[0].get_end_node().node.id, 11)

    def test_valid_node3(self):
        node = Scads.get_node_by_conceptnet_id("/c/en/toilet_tissue")
        self.assertEqual(node.get_datasets(), ['ImageNet'])
        images = node.get_images()
        self.assertEqual(len(images), 2)
        self.assertTrue('imagenet_1k/imagenet_1k_full/train/n15075141_53219.JPEG'
                        in images)

        # node2 = Scads.get_node_by_conceptnet_id("/c/en/bottle")
        # self.assertEqual(node2.get_datasets(), ['COCO2014'])
        # images = node2.get_images()
        # self.assertEqual(len(images), 1)
        # self.assertTrue('coco2014/coco2014_full/train/COCO_train2014_000000581674.jpg'
        #                 in images)
        #
        #
        # node6 = Scads.get_node_by_conceptnet_id("/c/en/bird")
        # self.assertEqual(node6.get_datasets(), ['VOC2009'])
        # images = node6.get_images()
        # self.assertEqual(len(images), 1)
        # self.assertTrue('voc2009/voc2009_full/train/2008_007250.jpg'
        #                 in images)
        # node5 = Scads.get_node_by_conceptnet_id("/c/en/tv_monitor")
        # self.assertEqual(node5.get_datasets(), ['VOC2009'])
        # images = node5.get_images()
        # self.assertEqual(len(images), 1)
        # self.assertTrue('voc2009/voc2009_full/test/2009_005240.jpg'
        #                 in images)


        # node3 = Scads.get_node_by_conceptnet_id("/c/en/person")
        # self.assertTrue('GoogleOpenImage' in node3.get_datasets())
        # images = node3.get_images()
        # self.assertEqual(len(images), 1)
        # self.assertTrue('google_open_image/google_open_image_full/test/067e21aeda713b53.jpg'
        #                 in images)
        #
        # node4 = Scads.get_node_by_conceptnet_id("/c/en/doll")
        # self.assertEqual(node4.get_datasets(), ['GoogleOpenImage'])
        # images = node4.get_images()
        # self.assertEqual(len(images), 1)
        # self.assertTrue('google_open_image/google_open_image_full/train/0100de671be66c38.jpg'
        #                 in images)



        node7 = Scads.get_node_by_conceptnet_id("/c/en/aircraft_carrier")
        print(node7.get_datasets())
        self.assertTrue('DomainNet: clipart' in node7.get_datasets())
        images = node7.get_images()
        self.assertEqual(len(images), 1)
        self.assertTrue('domainnet/domainnet_full/test/clipart_001_000005.jpg'
                        in images)

        node8 = Scads.get_node_by_conceptnet_id("/c/en/snake")
        self.assertTrue('DomainNet: clipart' in node8.get_datasets())
        images = node8.get_images()
        self.assertEqual(len(images), 1)
        self.assertTrue('domainnet/domainnet_full/train/clipart_270_000029.jpg'
                        in images)



    def test_undirected_relation(self):
        node = Scads.get_node_by_conceptnet_id("/c/en/zero")
        edges = node.get_neighbors()
        edges.sort(key=lambda x: x.get_relationship())
        edge = edges[0]

        self.assertEqual(edge.get_relationship(), "/r/Antonym")
        self.assertFalse(edge.is_directed())
        self.assertIsNotNone(edge.get_end_node())

    def test_directed_relation(self):
        node = Scads.get_node_by_conceptnet_id("/c/en/zero")
        edges = node.get_neighbors()
        edges.sort(key=lambda x: x.get_relationship())
        edge = edges[1]

        self.assertEqual(edge.get_relationship(), "/r/AtLocation")
        self.assertTrue(edge.is_directed())
        self.assertIsNotNone(edge.get_end_node())

    def test_neighbors(self):
        node = Scads.get_node_by_conceptnet_id("/c/en/zero")
        edges = node.get_neighbors()
        edges.sort(key=lambda x: x.get_relationship())

        neighbor1 = edges[0].get_end_node()
        self.assertEqual(neighbor1.node.id, 1)
        self.assertEqual(node.node.id, neighbor1.get_neighbors()[0].get_end_node().node.id)
        self.assertEqual(len(neighbor1.get_images()), 2)

        neighbor2 = edges[1].get_end_node()
        self.assertEqual(neighbor2.node.id, 2)
        self.assertEqual(neighbor2.get_datasets(), [])
        self.assertEqual(len(neighbor2.get_images()), 0)

    def test_weights(self):
        node = Scads.get_node_by_conceptnet_id("/c/en/zero")
        edges = node.get_neighbors()
        edges.sort(key=lambda x: x.get_relationship())
        self.assertEqual(edges[0].get_weight(), 2.5)
        self.assertEqual(edges[1].get_weight(), 1.0)

    @classmethod
    def tearDownClass(cls):
        Scads.close()
        os.remove(DB_PATH)


if __name__ == "__main__":
    unittest.main()
