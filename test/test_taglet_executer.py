import unittest
import numpy as np
from taglets.modules.module import TransferModule
from taglets.controller import Controller
from taglets.taglet_executer import TagletExecutor


class TestTagletExecuter(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Get Task
        controller = Controller()
        task = controller.get_task()[0]
        task.unlabeled_image_path = "../sql_data/MNIST/train"
        task.evaluation_image_path = "../sql_data/MNIST/test"
        batch_size = 32
        num_workers = 2
        cls.use_gpu = False
        cls.testing = False

        # Train and get Taglets
        module = TransferModule(task)
        train_data_loader, val_data_loader, train_image_names, train_image_labels = task.load_labeled_data(
            batch_size,
            num_workers)
        module.train_taglets(train_data_loader, val_data_loader, cls.use_gpu, "base", testing=True)
        cls.taglets = module.get_taglets()

        # Execute Taglets
        executor = TagletExecutor()
        executor.set_taglets(cls.taglets)
        cls.unlabeled_data_loader = task.load_unlabeled_data(batch_size, num_workers)[0]
        cls.label_matrix, cls.probabilities = executor.execute(cls.unlabeled_data_loader,
                                                                 cls.use_gpu,
                                                                 cls.testing)

    def test_soft_label_shape(self):
        self.assertTrue(self.label_matrix.shape[0] == len(self.unlabeled_data_loader.dataset))
        self.assertTrue(self.label_matrix.shape[1] == len(self.taglets))

    def test_probabilities_shape(self):
        self.assertTrue(len(self.probabilities) == len(self.unlabeled_data_loader.dataset))

    def test_soft_label_correctness(self):
        taglet_output = self.taglets[0].execute(self.unlabeled_data_loader,
                                                self.use_gpu,
                                                self.testing)
        self.assertTrue(np.array_equal(taglet_output, self.label_matrix[:, 0]))


if __name__ == "__main__":
    unittest.main()
