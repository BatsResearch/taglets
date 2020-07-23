import os
import unittest
from taglets.scads import ScadsEmbedding
import numpy as np


class TestScadsEmbedding(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        embedding_path = os.path.join(os.environ.get("TRAVIS_BUILD_DIR"), 'download', 'numberbatch-en-19.08.txt.gz')
        ScadsEmbedding.load(embedding_path)

    def test_oov(self):
        classes = ['/c/en/brown_university_wow',
                   '/c/en/ski_jump',
                   '/c/en/grotto',
                   '/c/en/swimming_pool_outdoor',
                   '/c/en/velodrome_outdoor',
                   '/c/en/newsstand_outdoor',
                   '/c/en/carport_freestanding',
                   '/c/en/residential_neighborhood',
                   '/c/en/airport_entrance']
        expected_classes = ['/c/en/brown_university',
                            '/c/en/ski_jump',
                            '/c/en/grotto',
                            '/c/en/swimming_pool',
                            '/c/en/velodrome',
                            '/c/en/newsstand',
                            '/c/en/carport',
                            '/c/en/residential_area',
                            '/c/en/airport_terminal']
        related_classes = [ScadsEmbedding.get_related_nodes(label, limit=1, is_node=False)[0] for label in classes]
        self.assertListEqual(related_classes, expected_classes)

    def test_related_nodes(self):
        related_classes = ScadsEmbedding.get_related_nodes('/c/en/dog', limit=100, is_node=False)
        self.assertTrue('/c/en/dog' in related_classes)
        self.assertTrue('/c/en/pet_animal' in related_classes)
        self.assertTrue('/c/en/household_pet' in related_classes)
        related_classes = ScadsEmbedding.get_related_nodes('/c/en/cat', limit=100, is_node=False)
        self.assertTrue('/c/en/cat' in related_classes)
        self.assertTrue('/c/en/pet_animal' in related_classes)
        self.assertTrue('/c/en/household_pet' in related_classes)
        related_classes = ScadsEmbedding.get_related_nodes('/c/en/bear', limit=200, is_node=False)
        self.assertTrue('/c/en/moose' in related_classes)
        
    def test_similarity(self):
        self.assertAlmostEqual(ScadsEmbedding.get_similarity('/c/en/top', '/c/en/top', is_node=False), 1)
        self.assertGreater(ScadsEmbedding.get_similarity('/c/en/apple', '/c/en/apple_juice', is_node=False), 0.5)
        self.assertGreater(ScadsEmbedding.get_similarity('/c/en/bear', '/c/en/brown_bear', is_node=False), 0.8)
        self.assertEqual(ScadsEmbedding.get_similarity('/c/en/bear', '/c/en/brown_bear', is_node=False),
                         ScadsEmbedding.get_similarity('/c/en/brown_bear', '/c/en/bear', is_node=False))
        
    def test_get_vector(self):
        vec = ScadsEmbedding.get_vector('/c/en/pen', is_node=False)
        self.assertAlmostEqual(np.linalg.norm(vec), 1)
    

if __name__ == "__main__":
    unittest.main()
