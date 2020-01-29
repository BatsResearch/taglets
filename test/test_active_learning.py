import unittest
import numpy as np
from modules.active_learning import LeastConfidenceActiveLearning, RandomActiveLearning


class TestRandomActiveLearning(unittest.TestCase):
    def setUp(self):
        """
        Set up variables that will be used for testing
        """
        self.unlabeled_images_names = ['0.png', '1.png', '2.png', '3.png', '4.png', '5.png', '6.png', '7.png',
                                       '8.png', '9.png', '10.png', '11.png', '12.png', '13.png', '14.png']
        self.random_active_learning = RandomActiveLearning()
        
    def test_once(self):
        """
        Test the active learning module by running it once
        """
        available_budget = 7
        candidates = self.random_active_learning.find_candidates(available_budget, self.unlabeled_images_names)
        self.assertEqual(len(candidates), available_budget)
        for candidate in candidates:
            self.assertIn(candidate, self.unlabeled_images_names)

    def test_three_times(self):
        """
        Test the active learning module as if we have three checkpoints
        """
        labeled_images_names = []
        available_budget = 5
        for i in range(3):
            rest_unlabeled_images_names = []
            for img in self.unlabeled_images_names:
                if img not in labeled_images_names:
                    rest_unlabeled_images_names.append(img)
        
            candidates = self.random_active_learning.find_candidates(available_budget, rest_unlabeled_images_names)
            labeled_images_names.extend(candidates)

            self.assertEqual(len(candidates), available_budget)
            for candidate in candidates:
                self.assertIn(candidate, rest_unlabeled_images_names)


class TestActiveLearning(unittest.TestCase):
    def setUp(self):
        """
        Set up variables that will be used for testing
        """
        self.unlabeled_images_names = ['0.png', '1.png', '2.png', '3.png', '4.png', '5.png', '6.png', '7.png',
                                       '8.png', '9.png', '10.png', '11.png', '12.png', '13.png', '14.png']
        self.confidence_active_learning = LeastConfidenceActiveLearning()
    
    def test_once(self):
        """
        Test the active learning module by running it once
        """
        available_budget = 5
        next_candidates = np.asarray([9, 14, 4, 6, 8, 3, 12, 5, 13, 10, 7, 1, 11, 0, 2])
        self.confidence_active_learning.set_candidates(next_candidates)
        candidates = self.confidence_active_learning.find_candidates(available_budget, self.unlabeled_images_names)
        self.assertEqual(candidates, ['9.png', '14.png', '4.png', '6.png', '8.png'])
        
    def test_three_times(self):
        """
        Test the active learning module as if we have three checkpoints
        """
        labeled_images_names = []
        available_budget = 5
        next_candidates = np.asarray([[9, 14, 4, 6, 8], [3, 8, 4, 9, 6], [3, 1, 4, 0, 2]])
        correct_order = ['9.png', '14.png', '4.png', '6.png', '8.png', '3.png', '12.png',
                        '5.png', '13.png', '10.png', '7.png', '1.png', '11.png', '0.png', '2.png']
        ct = 0
        for i in range(3):
            self.confidence_active_learning.set_candidates(next_candidates[i])
            
            rest_unlabeled_images_names = []
            for img in self.unlabeled_images_names:
                if img not in labeled_images_names:
                    rest_unlabeled_images_names.append(img)
                    
            candidates = self.confidence_active_learning.find_candidates(available_budget, rest_unlabeled_images_names)
            labeled_images_names.extend(candidates)
            
            self.assertEqual(candidates, correct_order[ct: ct+available_budget])
            ct += available_budget
    
    
if __name__ == '__main__':
    unittest.main()
