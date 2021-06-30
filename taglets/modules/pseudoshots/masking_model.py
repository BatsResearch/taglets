import torch.nn as nn
import torch

from .resnet12 import conv3x3, DropBlock, SELayer, BasicBlock


class Multimodule(nn.Module):
    def __init__(self, block, inplanes, channels, keep_prob=1.0, avg_pool=False, drop_rate=0.0,
                 dropblock_size=5, use_se=False, final_relu=True, max_pool=True):
        super(Multimodule, self).__init__()

        self.inplanes = inplanes
        self.use_se = use_se
        self.layers = nn.ModuleList()
        for channel in channels[:-1]:
            self.layers.append(self._make_layer(block, 1, channel,
                                       stride=2, drop_rate=drop_rate, drop_block=True, block_size=dropblock_size, max_pool=max_pool))
        self.layers.append(self._make_layer(block, 1, channels[-1],
                                   stride=2, drop_rate=drop_rate, drop_block=True, block_size=dropblock_size,
                                            final_relu=final_relu, max_pool=max_pool))
        self.out_dim = channels[-1]
        self.keep_prob = keep_prob
        self.dropout = nn.Dropout(p=1 - self.keep_prob, inplace=False)
        self.drop_rate = drop_rate

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, n_block, planes, stride=1, drop_rate=0.0, drop_block=False, block_size=1, final_relu=True, max_pool=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        if n_block == 1:
            layer = block(self.inplanes, planes, stride, downsample, drop_rate, drop_block, block_size, self.use_se, final_relu=final_relu, max_pool=max_pool)
        else:
            layer = block(self.inplanes, planes, stride, downsample, drop_rate, self.use_se, final_relu=final_relu, max_pool=max_pool)
        layers.append(layer)
        self.inplanes = planes * block.expansion

        for i in range(1, n_block):
            if i == n_block - 1:
                layer = block(self.inplanes, planes, drop_rate=drop_rate, drop_block=drop_block,
                              block_size=block_size, use_se=self.use_se, final_relu=final_relu, max_pool=max_pool)
            else:
                layer = block(self.inplanes, planes, drop_rate=drop_rate, use_se=self.use_se, final_relu=final_relu, max_pool=max_pool)
            layers.append(layer)

        return nn.Sequential(*layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x


class MultimoduleMasking(nn.Module):
    def __init__(self, maskingmodule, inplanes, channels, final_relu, activation, max_pool, **kwargs):
        super().__init__()

        self.masking_model = maskingmodule(inplanes=inplanes, channels=channels, final_relu=final_relu, max_pool=max_pool, **kwargs)
        self.out_dim = self.masking_model.out_dim
        if activation == 'sigmoid':
            self.act_func = nn.Sigmoid()
        elif activation == 'softmax':
            self.act_func = nn.Softmax(dim=0)
        elif activation == 'linear':
            self.act_func = None
        else:
            raise ValueError('Invalid Activation Function.')

    def forward(self, x):
        mask = self.masking_model(x)
        if self.act_func is not None:
            mask = self.act_func(mask)
        return mask


def multi_block(**kwargs):
    return Multimodule(BasicBlock, **kwargs)


def multi_block_masking(**kwargs):
    return MultimoduleMasking(multi_block, **kwargs)


class MaskingHead(nn.Module):
    def __init__(self, encoder_dim, masking_module_args):
        super().__init__()
        masking_module_args['inplanes'] = encoder_dim * 2
        self.masking_model = multi_block_masking(**masking_module_args)

    def forward(self, data_dict):
        # note: masking-layer assumes that image encodings are given as input
        embedding = data_dict['embed']
        batch_shape = embedding.shape[:1]
        mask = self.masking_model(embedding)
        mask = mask.view(*batch_shape, *mask.shape[1:])
        return {'mask': torch.mul(embedding, mask)}