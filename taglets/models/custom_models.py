from torchvision.models.resnet import ResNet, BasicBlock
import torch
import torch.nn as nn


class MnistResNet(ResNet):
    """
    A custom ResNet model used in a Taglet.
    """
    def __init__(self):
        """
        Create a new MnistResNet model.
        """
        super(MnistResNet, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes=10)
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    def forward(self, x):
        """
        The forward pass of the model.
        :param x: The input data
        :return: The logits output
        """
        return super(MnistResNet, self).forward(x)


class Linear(nn.Module):
    def __init__(self, in_feature=64, out_feature=10):
        super().__init__()
        self.classifier = nn.Sequential(nn.Linear(in_feature, out_feature))

    def forward(self, x):
        x = self.classifier(x)
        return x


class ConvEncoder(nn.Module):
    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            self.conv_block(x_dim, hid_dim),
            self.conv_block(hid_dim, hid_dim),
            self.conv_block(hid_dim, hid_dim),
            self.conv_block(hid_dim, z_dim),
        )

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)

    def conv_block(self, in_channels, out_channels):
        bn = nn.BatchNorm2d(out_channels)
        nn.init.uniform_(bn.weight)  # For pytorch 1.2 or later
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            bn,
            nn.ReLU(),
            nn.MaxPool2d(2)
        )