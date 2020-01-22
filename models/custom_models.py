from torchvision.models.resnet import ResNet, BasicBlock
import torch


class MnistResNet(ResNet):
    """
    A custom ResNet model used in a Taglet.
    """
    def __init__(self):
        """
        Create a new MnistResNet model.
        """
        super(MnistResNet, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes=10)
        self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    def forward(self, x):
        """
        The forward pass of the model.
        :param x: The input data
        :return: The softmax output
        """
        return torch.softmax(super(MnistResNet, self).forward(x), dim=-1)
