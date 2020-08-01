import os
import unittest 
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from taglets.task import Task
from taglets.modules.zsl_kg_lite import ZSLKGModule
from taglets.modules.zsl_kg_lite.example_encoders.resnet import ResNet
from taglets.data.custom_dataset import CustomDataset

TEST_DATA = os.path.dirname(os.path.realpath(__file__))


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
        cls.module = ZSLKGModule(task)

        options = cls.module.taglets[0].options


        cls.adj_lists = {
            0: [[1, 12, 1.0]],
            1: [[0, 12, 0.51], [0, 11, 0.49]],
            2: [[1, 12, 1.0]]
        }
        device = torch.device('cpu')
        features = torch.rand((3, 300))


        # the graph neural network model
        cls.taglet = cls.module.taglets[0]
        cls.model = cls.module.taglets[0]._get_model(features, cls.adj_lists, device, options)

        cls.resnet = ResNet()

        # load sample images for classification
        
        image_path = [
            os.path.join(TEST_DATA, '../../test/test_data/scads/imagenet_1k/imagenet_1k_full/test/ILSVRC2012_val_00049419.JPEG'),
            os.path.join(TEST_DATA, '../../test/test_data/scads/imagenet_1k/imagenet_1k_full/test/ILSVRC2012_val_00049991.JPEG')
        ]

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

        _transforms = transforms.Compose([transforms.Resize(224),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        normalize])
        dataset = CustomDataset(image_path, transform=_transforms)
        cls.loader = DataLoader(dataset=dataset, batch_size=32,
                        shuffle=False, num_workers=2)
    
    def test_dummy(self):
        self.assertTrue(2, 2)

    def test_check_model_load(self):
        """Checking if model loaded correctly and checking the output vector shape
        """
        self.assertTrue(self.model.label_dim, 2049)
        self.assertTrue(self.model.gnn_modules[-1].w.size(0), 2048)
        self.assertTrue(self.model.gnn_modules[-1]([1]).size(1), 2049)
        self.assertTrue(self.model([0, 1]).size(1), 2049)
    
    def test_check_example_encoder(self):
        """checks the output dimensions of the resnet loader
        """
        self.assertTrue(self.resnet)
    
    def test_predict(self):
        class_rep = self.model([0, 2])
        self.resnet.eval()
        self.model.eval()
        preds = self.taglet._predict(self.loader, self.resnet, class_rep)
        self.assertTrue(len(preds), 2)


if __name__ == '__main__':
    unittest.main()