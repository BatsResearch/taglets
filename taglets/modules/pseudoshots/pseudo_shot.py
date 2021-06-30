from ..module import Module
from ...pipeline import Cache, ImageTaglet
from enum import IntEnum


class Encoder(IntEnum):
    RESNET_50 = 0

    # X1 = single width, X2 = double width (https://github.com/google-research/simclr)
    SIMCLR_50_X1 = 1
    SIMCLR_50_X2 = 2


class PSModule(Module):
    def __init__(self, task):
        super().__init__(task)
        ps_args = {}

        # few-shot params
        ps_args['n_shot'] = 1
        ps_args['n_way'] = 15
        ps_args['n_query'] = 15
        ps_args['n_pseudo'] = 15

        ps_args['img_encoder_type'] = Encoder.RESNET_50
        ps_args['lr'] =

        self.taglets = [PSTaglet(task, **ps_args)]


class PSTaglet(ImageTaglet):
    def __init__(self, task, **kwargs):
        super().__init__(task)

        # training only parameters
        self.n_shot = kwargs.get('n_shot', 5)
        self.n_way = kwargs.get('n_way', 15)
        self.n_query = kwargs.get('n_query', 15)

        self.n_pseudo = kwargs.get('n_pseudo', 15)

        self.img_encoder_type = kwargs.get('img_encoder_type', Encoder.RESNET_50)
        self.lr = kwargs.get('lr', self.lr)

    def train(self, train_data_loader, val_data_loader, unlabeled_data_loader=None):
        pass

    def _train(self):
        # Pretrain Masking Module
        pass