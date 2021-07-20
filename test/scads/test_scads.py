import unittest
import os
from taglets.scads import Scads
from taglets.scads.create.install import Installer, CifarInstallation, MnistInstallation, ImageNetInstallation, COCO2014Installation, GoogleOpenImageInstallation, VOC2009Installation, DomainNetInstallation, HMDBInstallation, UCF101Installation, MarsSurfaceInstallation, MslCuriosityInstallation

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
DOMAINNET_PATH = "domainnet"
HMDB_PATH = "hmdb"
UCF101_PATH = "ucf101"
MARS_SURFACE_PATH = "mars_surface_imgs"
MSL_CURIOSITY_PATH = "msl_curiosity_imgs"

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
        installer.install_dataset(ROOT, DOMAINNET_PATH, DomainNetInstallation('clipart'))
        installer.install_dataset(ROOT, HMDB_PATH, HMDBInstallation())
        installer.install_dataset(ROOT, UCF101_PATH, UCF101Installation())
        installer.install_dataset(ROOT, MARS_SURFACE_PATH, MarsSurfaceInstallation())
        installer.install_dataset(ROOT, MSL_CURIOSITY_PATH, MslCuriosityInstallation())

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
        self.assertEqual(len(images), 3)
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

    def test_mars(self):
        # MSL curiosity
        node = Scads.get_node_by_conceptnet_id("/c/en/tray")
        print(node.get_datasets())
        self.assertTrue('MarsSurface' in node.get_datasets())
        images = node.get_images()
        self.assertEqual(len(images), 5)
        self.assertTrue('mars_surface_imgs/mars_surface_imgs_full/test/0572ML0023150000204990I01_DRCL.JPG'
                        in images)


    def test_clips(self):
        """"""  


        class_label = "/c/en/run"
        # HMDB
        node = Scads.get_node_by_conceptnet_id(class_label)
        self.assertEqual(node.node.id, 29)
        self.assertEqual(node.get_conceptnet_id(), class_label)

        self.assertEqual(node.get_datasets(images=False), ['HMDB'])

        images = node.get_images()
        self.assertEqual(len(images), 0)
        clips = node.get_clips()
        self.assertEqual(len(clips), 19)
        self.assertTrue('hmdb/hmdb_full/test'
                        in [x[0] for x in clips])
        clip = clips[0]
        self.assertEqual(clip[0], "hmdb/hmdb_full/train")
        self.assertEqual(clip[1], 231465)
        self.assertEqual(clip[2], 231565)

        # UCF101
        class_label_2 = "/c/en/jet_ski"
        node1 = Scads.get_node_by_conceptnet_id(class_label_2)
        self.assertEqual(node1.node.id, 30)
        self.assertEqual(node1.get_conceptnet_id(), class_label_2)

        self.assertEqual(node1.get_datasets(images=False), ['UCF101'])

        images = node1.get_images()
        self.assertEqual(len(images), 0)
        clips = node1.get_clips()
        self.assertEqual(len(clips), 100)
        self.assertTrue('ucf101/ucf101_full/test'
                        in [x[0] for x in clips])
        clip = clips[0]
        self.assertEqual(clip[0], "ucf101/ucf101_full/train")
        self.assertEqual(clip[1], 23470)
        self.assertEqual(clip[2], 23803)

        # Multi-nodes
        class_label_3 = "/c/en/playing_dhol"

        try:
            node_multi = Scads.get_node_by_conceptnet_id(class_label_3)
            clips = node_multi.get_clips()
        except:
            # Get single concepts
            label = class_label_3.split('/')[-1]
            nodes = [f"/c/en/{w.strip()}" for w in label.split('_')]
            objects_node = [Scads.get_node_by_conceptnet_id(n) for n in nodes]

            # Write the OR query to execute only one query for the label
            query = ' '.join([f'nodes.id = {obj.node.id} OR' \
                  for obj in objects_node[:-1]])
            query = ' '.join([query, f'nodes.id = {objects_node[-1].node.id}'])

            # Execute the query
            clips = objects_node[0].get_clips_multiple(query)
            self.assertEqual(len(clips), 1267)
        
            
            
        




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
        self.assertEqual(len(neighbor1.get_images()), 3)

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
