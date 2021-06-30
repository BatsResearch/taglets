import torch
import logging

log = logging.getLogger(__name__)

from accelerate import Accelerator
from ..module import Module
from ...pipeline import Cache, ImageTaglet
from enum import IntEnum
from .masking_model import MaskingHead, MultimoduleMasking
from .utils import freeze_module, get_total_size

accelerator = Accelerator()


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
        ps_args['masking_ckpt_path'] = 'predefined/pseudoshots/masking_module.pt'
        ps_args['masking_args'] = {
            'channels': [640, 320, 1],
            'final_relu': False,
            'max_pool': False,
            'activation': 'sigmoid',
            'dropblock_size': 5
        }

        self.taglets = [PSTaglet(task, **ps_args)]


class PSTaglet(ImageTaglet):
    def __init__(self, task, masking_ckpt_path, masking_args, **kwargs):
        super().__init__(task)

        # training only parameters
        self.n_shot = kwargs.get('n_shot', 5)
        self.n_way = kwargs.get('n_way', 15)
        self.n_query = kwargs.get('n_query', 15)

        self.n_pseudo = kwargs.get('n_pseudo', 15)

        img_encoder_type = kwargs.get('img_encoder_type', Encoder.RESNET_50)
        self.img_encoder = self._set_img_encoder(img_encoder_type)
        self.masking_module = self._load_masking_module(masking_ckpt_path, **masking_args)

        self.lr = kwargs.get('lr', self.lr)
        self._params_to_update = []

        backbone = torch.nn.Sequential(*list(self.img_encoder.children())[:-1])
        im_encoder_shape = self._get_model_output_shape(self.task.input_shape, backbone)
        self.support_embeddings = torch.zeros((len(self.task.classes), get_total_size(im_encoder_shape)))


    @staticmethod
    def _load_masking_module(ckpt_path, **kwargs):
        # TODO: determine how to sync encoder dim and masking module dim
        masking_module = MultimoduleMasking(**kwargs)
        masking_module.load_state_dict(torch.load(ckpt_path))
        masking_module.eval()
        return masking_module

    def _set_img_encoder(self, img_encoder_type):
        if img_encoder_type == Encoder.RESNET_50:
            return self.model
        elif img_encoder_type == Encoder.SIMCLR_50_X1:
            # TODO: load from file
            return None
        elif img_encoder_type == Encoder.SIMCLR_50_X2:
            # TODO: load from file
            return None
        else:
            self.valid = False
            raise ValueError(f'{img_encoder_type} is not a valid image encoder')

    def train(self, train_data_loader, val_data_loader, unlabeled_data_loader=None):
        """
        Note: train is somewhat of a misnomer because the nearest neighbor classifier is parameter-free.

        Steps:
        1. Iterate over each training example.
        2. Compute the encoding of each training example and save it to a list.
        3. For each class, compute
        """

        main_dev = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        with torch.no_grad():
            if accelerator.is_local_main_process:
                img_encoder = self.img_encoder.to(main_dev)
                masking_module = self.masking_module.to(main_dev)
                self.support_embeddings = self.support_embeddings.to(main_dev)

                # first comp = existence bit; second comp = hit count
                cls_matrix = torch.zeros((self.support_embeddings.shape[0], 2), dtype=torch.int32)
                for x, y in train_data_loader:
                    x, y = x.to(main_dev), y.to(main_dev)
                    x_embeds = img_encoder(x)

                    for i in range(self.support_embeddings.shape[0]):
                        cls_mask = torch.where(y == i, True, False)
                        masked_embeds = x_embeds[cls_mask]

                        if masked_embeds.nelement() > 0:
                            self.support_embeddings[i] += x_embeds[cls_mask].sum(dim=0).reshape(-1)
                            cls_matrix[i, 0] = 1
                            cls_matrix[i, 1] += masked_embeds.shape[0]

                if cls_matrix[:, 0].sum() != cls_matrix.shape[0]:
                    self.valid = False
                    log.error('FSL ERROR: Number of classes in dataset not equal to number of task classes.')
                    return

                # normalize summations
                counts = cls_matrix[:, 1].reshape(-1)
                self.support_embeddings *= torch.where(counts > 0, 1.0 / counts, 1.0)

    def _train(self):
        # Pretrain Masking Module
        pass