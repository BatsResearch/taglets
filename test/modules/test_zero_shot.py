import unittest
import torch

from taglets.scads import Scads
from taglets.scads.create.install import Installer, MnistInstallation
from taglets.controller import Controller
from taglets.task import Task
from taglets.modules.zero_shot import ZeroShotModule, ZeroShotTaglet
from taglets.modules.zsl_kg_lite.example_encoders.resnet import ResNet

class TestZeroShotModule(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Get Task
        classes = [
            'sheep',
            'walrus'
        ]

        task = Task('zero-shot', classes, (224, 224, 3),
                    labeled_train_data=None, unlabeled_train_data=None,
                    validation_data=None, whitelist=None,
                    scads_path=None)

        self.adj_lists = {
            0: [[1, 12, 1.0]],
            1: [[0, 12, 0.51], [0, 11, 0.49]],
            2: [[1, 12, 1.0]]
        }

        device = torch.device('cpu')

        features = torch.rand((3, 300))

        module = ZeroShotModule(task)

        options = module.taglets[0].options

        # the graph neural network model
        cls.model = module.taglets[0]._get_model(features, adj_lists, device, options)

        cls.resnet = ResNet()

        # load sample images for classification
    
    def test_dummy(self):
        self.assertTrue(2, 2)

    def test_check_model_load(self):
        """Checking if model loaded correctly and checking the output vector shape
        """
        self.assertTrue(self.model.label_dim, 2049)
        self.assertTrue(self.model.gnn_modules[-1].w.size(0), 2048)
        self.assertTrue(self.model.gnn_modules[-1]([1]).size(1), 2049)
        self.assertTrue(self.model([1]).size(1), 2049)
    
    def test_check_example_encoder(self):
        """checks the output dimensions of the resnet loader
        """
        self.assertTrue(self.resnet)
    
    def test_predict(self):
        # change graph
        
        # 

if __name__ == '__main__':
    unittest.main()