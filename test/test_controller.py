import unittest
from controller import Controller


class TestController(unittest.TestCase):
    def setUp(self):
        self.controller = Controller()

    def test_labeled_increasing(self):
        '''
        Tests that the number of labeled data points is increasing at each checkpoint.
        '''
        current = len(self.controller.task.labeled_images)
        for i in range(self.controller.num_base_checkpoints):
            self.controller.run_one_checkpoint(i)
            new = len(self.controller.task.labeled_images)
            self.assertTrue(new > current, "Number of labeled data points is not increasing.")
            current = new

    def test_unlabeled_decreasing(self):
        '''
        Tests that the number of unlabeled data points is decreasing at each checkpoint.
        '''
        current = len(self.controller.task.get_unlabeled_image_names())
        for i in range(self.controller.num_base_checkpoints):
            self.controller.run_one_checkpoint(i)
            new = len(self.controller.task.get_unlabeled_image_names())
            self.assertTrue(new < current, "Number of unlabeled data points is not decreasing.")
            current = new

    def constant_total(self):
        '''
        Tests that the total number of data points (labeled and unlabeled) is constant at each checkpoint.
        :return:
        '''
        current = len(self.controller.task.labeled_images) + len(self.controller.task.get_unlabeled_image_names())
        for i in range(self.controller.num_base_checkpoints):
            self.controller.run_one_checkpoint(i)
            new = len(self.controller.task.labeled_images) + len(self.controller.task.get_unlabeled_image_names())
            self.assertTrue(new == current, "Number of unlabeled data points is not decreasing.")
