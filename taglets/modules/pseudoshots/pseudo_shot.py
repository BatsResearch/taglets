import torch
import logging
import random
import os
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import copy 
import torchvision.models as models


log = logging.getLogger(__name__)

from ...data.custom_dataset import CustomImageDataset, PseudoshotImageDataset
from accelerate import Accelerator
from ..module import Module
from ...pipeline import Cache, ImageTaglet
from enum import IntEnum
from .masking_model import MaskingHead, multi_block_masking
from .utils import get_total_size
from ...scads import Scads, ScadsEmbedding

from .resnet12 import resnet12
from .simclr import get_resnet


accelerator = Accelerator()


class Encoder(IntEnum):
    RESNET_50 = 0

    # X1 = single width (https://github.com/google-research/simclr)
    SIMCLR_50_X1 = 1


class Metric(IntEnum):
    COSINE = 0
    SQRT = 1
    DOT = 2


class NearestNeighborClassifier(nn.Module):
    def __init__(self, cls_prototypes, encoder, metric=Metric.COSINE):
        super().__init__()
        self.cls_prototypes = cls_prototypes
        self.encoder = encoder
        self.metric = metric

    def set_prototypes(self, proto):
        self.cls_prototypes = proto

    def forward(self, x):
        """
        x: (batch, c, h, w) -[encoder]-> (batch, encoding_dim) -[NN Classifier]-> (batch, logits)
        """
        x = self.encoder(x)
        cls_prototypes = self.cls_prototypes

        if self.metric == Metric.COSINE:
            cls_prototypes = F.normalize(cls_prototypes, dim=-1)
            x = F.normalize(x, dim=-1)

        logits = None
        
        x = x.reshape((x.shape[0], -1))
        if x.dim() == 2:
            if self.metric == Metric.DOT:
                logits = torch.mm(x, cls_prototypes.t())
            elif self.metric == Metric.COSINE:
                a = F.normalize(x, dim=-1).type(torch.FloatTensor)
                b = F.normalize(cls_prototypes, dim=-1).t().type(torch.FloatTensor)
                logits = torch.matmul(a, b)
            elif self.metric == Metric.SQRT:
                logits = -(x.unsqueeze(1) -
                           cls_prototypes.unsqueeze(0)).pow(2).sum(dim=-1)
        elif x.dim() == 3:
            if self.metric == Metric.DOT:
                logits = torch.bmm(x, cls_prototypes.permute(0, 2, 1))
            elif self.metric == Metric.COSINE:
                logits = torch.bmm(F.normalize(x, dim=-1),
                                   F.normalize(cls_prototypes, dim=-1).permute(0, 2, 1))
            elif self.metric == Metric.SQRT:
                logits = -(self.x.unsqueeze(2) -
                           cls_prototypes.unsqueeze(1)).pow(2).sum(dim=-1)
        else:
            raise ValueError('Too many dims!')
        return logits.to(x.device)


class PseudoShotModule(Module):
    def __init__(self, task):
        super().__init__(task)
        ps_args = {}

        # few-shot params; only necessary if you're training 
        ps_args['n_shot'] = 1
        ps_args['n_way'] = 15
        ps_args['n_query'] = 15
        ps_args['n_pseudo'] = 15

        ps_args['img_encoder_type'] = Encoder.RESNET_50
        ps_args['img_encoder_ckpt_path'] = None
        ps_args['masking_ckpt_path'] = 'predefined/pseudoshots/resnet50_train_val_test_224_1shot.pth'
        ps_args['masking_args'] = {
            'channels': [640, 320, 1],
            'final_relu': False,
            'max_pool': False,
            'activation': 'sigmoid',
            'dropblock_size': 5,
            'inplanes': 4096
        }

        ps_args['metric'] = Metric.COSINE
        self.taglets = [PseudoShotTaglet(task, **ps_args)]


class PseudoShotTaglet(ImageTaglet):
    def __init__(self, task, masking_ckpt_path, masking_args, **kwargs):
        super().__init__(task)

        # only for development purposes
        self.dev_test = False
        self.dev_shape = (3, 84, 84)

        self.n_shot = kwargs.get('n_shot', 5)

        # training only parameters
        self.n_way = kwargs.get('n_way', 15)
        self.n_query = kwargs.get('n_query', 15)

        # PS budget
        self.max_real_image = 6
        self.img_per_related_class = 5 if not os.environ.get("CI") else 1
        self.num_related_class = 3

        if not os.environ.get('CI'):
            self.n_pseudo = self.max_real_image + self.img_per_related_class * self.num_related_class
        else:
            self.n_pseudo = 1

        # mask batching for memory issues
        self.mask_batch_size = 20

        img_encoder_type = kwargs.get('img_encoder_type', Encoder.RESNET_50)

        # load image encoder and masking module 
        self.img_encoder = self._set_img_encoder(img_encoder_type)
        self.masking_module = self._load_masking_module(masking_ckpt_path, **masking_args)
        self.masking_head = MaskingHead(self.masking_module)

        self.lr = kwargs.get('lr', self.lr)
        self._params_to_update = []

        backbone = torch.nn.Sequential(*list(self.img_encoder.children())[:-1])
        #im_encoder_shape = self._get_model_output_shape(self.dev_shape, backbone)

        out_dim = 640 * 5 * 5 if self.dev_test else 2048 * 7 * 7
        self.support_embeddings = torch.zeros((len(self.task.classes), out_dim))

        if os.getenv("LWLL_TA1_PROB_TASK") is not None:
            self.test_path = '/home/tagletuser/trained_models/pseudoshots'
        else:
            self.test_path = 'trained_models/pseudoshots'
        if not os.path.exists(self.test_path):
            os.makedirs(self.test_path)

        self.model = NearestNeighborClassifier(None, self.img_encoder, kwargs.get('metric', Metric.COSINE))
        self.proto_file = os.path.join(self.test_path, f'test_protos.pth')

    def transform_image(self):
        """
        Get the transform to be used on an image.
        :return: A transform
        """
        data_mean = [0.485, 0.456, 0.406]
        data_std = [0.229, 0.224, 0.225]

        if self.dev_test:
            return transforms.Compose([
                transforms.Resize(self.dev_shape[1:]),
                transforms.ToTensor(),
                transforms.Normalize(mean=data_mean, std=data_std)
            ])
        return transforms.Compose([
                transforms.Resize(self.task.input_shape),
                transforms.ToTensor(),
                transforms.Normalize(mean=data_mean, std=data_std)
            ])

    def _get_scads_data(self):
        data = Cache.get("pseudoshots", self.task.classes)
        if data is not None:
            image_paths, image_labels, all_related_class, label_mask = data
        else:
            root_path = Scads.get_root_path()
            Scads.open(self.task.scads_path)
            ScadsEmbedding.load(self.task.scads_embedding_path)
            image_paths = []
            image_labels = []
            
            label_mask = []
            visited = set()

            def get_images(node, label, budget, is_real):
                if node.get_conceptnet_id() not in visited:
                    visited.add(node.get_conceptnet_id())
                    images = node.get_images_whitelist(self.task.whitelist)
                    sample_cnt = min(budget, len(images), self.img_per_related_class)
                    if sample_cnt == 0:
                        return False, budget

                    budget -= sample_cnt
                    images = random.sample(images, sample_cnt)
                    images = [os.path.join(root_path, image) for image in images]
                    image_paths.extend(images)
                    image_labels.extend([label] * len(images))
                    label_mask.extend([is_real] * len(images))

                    log.debug("Source class found: {}".format(node.get_conceptnet_id()))
                    return True, budget
                return False, budget

            all_related_class = 0
            aux_budget = self.n_pseudo
            for i, conceptnet_id in enumerate(self.task.classes):
                cls_label = i
                cur_related_class = 0

                target_node = Scads.get_node_by_conceptnet_id(conceptnet_id)

                # prioritize data of the ame class
                valid, aux_budget = get_images(target_node, cls_label, self.max_real_image, 
                                                                       is_real=1)
                if valid:
                    cur_related_class += 1
                    all_related_class += 1

                neighbors = ScadsEmbedding.get_related_nodes(target_node)
                for neighbor in neighbors:
                    valid, aux_budget = get_images(neighbor, cls_label, self.img_per_related_class, is_real=0)
                    if valid:
                        cur_related_class += 1
                        all_related_class += 1
                        if cur_related_class >= self.num_related_class:
                            break

                if aux_budget > 0:
                    logging.warning('aux budget for %s is positive.' % conceptnet_id)

            Scads.close()
            Cache.set('pseudoshots', self.task.classes,
                      (image_paths, image_labels, all_related_class, label_mask))

        transform = self.transform_image()
        train_dataset = PseudoshotImageDataset(image_paths,
                                           labels=image_labels,
                                           label_mask=label_mask,
                                           transform=transform)
        return train_dataset, all_related_class

    def _load_masking_module(self, ckpt_path, **kwargs):
        if self.dev_test:
            args = {'channels': [640, 320, 1],
                    'final_relu': False,
                    'max_pool': False,
                    'activation': 'sigmoid',
                    'dropblock_size': 5,
                    'inplanes': self.img_encoder.out_dim * 2}
            mask_path = 'predefined/pseudoshots/resnet12_mask.pth'
            masking_module = multi_block_masking(**args)
            masking_module.load_state_dict(torch.load(mask_path)['multi-block-masking_sd'])
            masking_module.eval()
            return masking_module

        masking_module = multi_block_masking(**kwargs)
        sd = dict(torch.load(ckpt_path)['model_sd'])
        new_sd = {}
        for k, v in sd.items():
            if 'encoder.masking_model' in k:
                new_sd[k.split('encoder.masking_model.', 1)[1]] = v
        masking_module.load_state_dict(new_sd)
        masking_module.eval()
        return masking_module

    def _set_img_encoder(self, img_encoder_type, ckpt_file=None):
        if self.dev_test:
            encoder = resnet12(**{'avg_pool': False, 'drop_rate': 0.1, 'dropblock_size': 5})
            encoder.load_state_dict(torch.load('predefined/pseudoshots/resnet12.pth')['resnet12_sd'])
        elif img_encoder_type == Encoder.RESNET_50:
            encoder = models.resnet50(pretrained=True)
            # remove head 
            encoder = nn.Sequential(*list(encoder.children())[:-2])
        elif img_encoder_type == Encoder.SIMCLR_50_X1:
            encoder = get_resnet()
            encoder.load_state_dict(torch.load('predefined/pseudoshots/simclr50.pth')['resnet'])
            encoder = nn.Sequential(*list(encoder.children())[:-1])
        else:
            self.valid = False
            raise ValueError(f'{img_encoder_type} is not a valid image encoder')
        encoder.eval()
        return encoder

    @staticmethod
    def _group_encodings(encodings, labels, cls_mat, group_mat):
        for i in range(cls_mat.shape[0]):
            cls_mask = labels == i
            masked_embeds = encodings[cls_mask]

            if masked_embeds.nelement() > 0:
                # add to class prototype & update hit and count
                group_mat[i] += encodings[cls_mask].sum(dim=0).reshape(-1)
                cls_mat[i, 0] = 1
                cls_mat[i, 1] += masked_embeds.shape[0]
        return cls_mat, group_mat


    def train(self, train_data, val_data, unlabeled_data=None):
        """
        Note: train is somewhat of a misnomer because the nearest neighbor classifier is 
        parameter-free.

        Steps:
        1. Iterate over each training example.
        2. Compute the encoding of each training example and save it to a list.
        3. For each class, compute the corresponding prototype using mean masked pseudoshot 
           embeddings and support embeddings.
        """
        log.info('Beginning training')

        # compute prototypes only on 
        main_dev = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        if accelerator.is_local_main_process:
            with torch.set_grad_enabled(False):
                if self.dev_test:
                    train_data = copy.deepcopy(train_data)
                    train_data.transform = self.transform_image()

                img_encoder = self.img_encoder.to(main_dev)
                masking_head = self.masking_head.to(main_dev)
                support_embeddings = self.support_embeddings.to(main_dev)
                train_data_loader = torch.utils.data.DataLoader(
                    dataset=train_data, batch_size=self.batch_size, shuffle=False,
                    num_workers=self.num_workers, pin_memory=True)

                # first comp = existence bit; second comp = hit count
                supp_cls_matrix = torch.zeros((support_embeddings.shape[0], 2), dtype=torch.int32).to(main_dev)
                for batch in train_data_loader:
                    x, y = batch[0], batch[1]
                    x, y = x.to(main_dev), y.to(main_dev)
                    x_embeds = img_encoder(x)

                    supp_cls_matrix, support_embeddings = PseudoShotTaglet._group_encodings(x_embeds,
                                                                                            y,
                                                                                            supp_cls_matrix,
                                                                                            support_embeddings)
                if supp_cls_matrix[:, 0].sum() != supp_cls_matrix.shape[0]:
                    self.valid = False
                    return

                # threshold to convert counts, represented as floats, to ints 
                eps = 0.5
                scads_train_data, all_related_class = self._get_scads_data()
                if all_related_class > 0:
                    scads_data_loader = torch.utils.data.DataLoader(
                        dataset=scads_train_data, batch_size=1, shuffle=False,
                        num_workers=self.num_workers, pin_memory=True)

                    aux_embeddings = torch.zeros(support_embeddings.shape).to(main_dev)
                    aux_cls_matrix = torch.zeros(supp_cls_matrix.shape, dtype=torch.int32).to(main_dev)

                    for x, y, mask in scads_data_loader:
                        x, y, mask = x.to(main_dev), y.to(main_dev), mask.to(main_dev).type(torch.int32)
                        x_embeds = img_encoder(x)

                        # should be bitmask
                        assert y[mask == 1].shape[0] + y[mask == 0].shape[0] == x.shape[0]

                        new_supp_examples = x_embeds[mask == 1]
                        new_supp_labels = y[mask == 1]
                        supp_cls_matrix, support_embeddings = PseudoShotTaglet._group_encodings(new_supp_examples,
                                                                                                new_supp_labels,
                                                                                                supp_cls_matrix,
                                                                                                support_embeddings)

                        aux_examples = x_embeds[mask == 0]
                        aux_labels = y[mask == 0]
                        aux_cls_matrix, aux_embeddings = PseudoShotTaglet._group_encodings(aux_examples,
                                                                                           aux_labels,
                                                                                           aux_cls_matrix,
                                                                                           aux_embeddings)
                    aux_counts = aux_cls_matrix[:, 1].reshape(-1).type(torch.DoubleTensor).to(main_dev)
                    norm_factor = torch.where(aux_counts > eps, 1.0 / aux_counts, 1.0)
                    norm_factor = norm_factor.unsqueeze(-1).expand(aux_embeddings.shape)
                    norm_aux_embeddings = aux_embeddings * norm_factor

                    supp_counts = supp_cls_matrix[:, 1].reshape(-1).type(torch.DoubleTensor).to(main_dev)
                    norm_factor = torch.where(supp_counts > eps, 1.0 / supp_counts, 1.0)
                    
                    # expand norms across row so that norm_factor.shape == support_embeddings.shape
                    norm_factor = norm_factor.unsqueeze(-1).expand(support_embeddings.shape)
                    norm_support_embeddings = support_embeddings * norm_factor 

                    # generate masks
                    log.debug(norm_support_embeddings.shape)
                    log.debug(norm_aux_embeddings.shape)

                    joint_embeddings = torch.cat((norm_support_embeddings, norm_aux_embeddings), dim=1)

                    log.debug(joint_embeddings.shape)
                    joint_embeddings = joint_embeddings.type(torch.FloatTensor).to(main_dev)

                    n_batches = joint_embeddings.shape[0] // self.mask_batch_size
                    assert joint_embeddings.shape[0] == aux_embeddings.shape[0]
                    for i in range(max(n_batches, 1)):
                        upper = min((i + 1) * self.mask_batch_size, joint_embeddings.shape[0])
                        embed = joint_embeddings[i * self.mask_batch_size: upper].to(main_dev)
                        pseudo = aux_embeddings[i * self.mask_batch_size: upper].to(main_dev)

                        # only use masking module for classes with aux data
                        idx_mask = (aux_counts[i * self.mask_batch_size: upper] > eps) & (supp_counts[i * self.mask_batch_size: upper] > eps)

                        if embed[idx_mask].shape[0] == 0:
                            continue 

                        if self.dev_test:
                            embed_shape = (-1, 1280, 5, 5)
                            pseudo_shape = (-1, 640, 5, 5)
                        else:
                            embed_shape = (-1, 4096, 7, 7)
                            pseudo_shape = (-1, 2048, 7, 7)
                        masked_embed = masking_head({'embed': embed[idx_mask].reshape(embed_shape), 'pseudo': pseudo[idx_mask].reshape(pseudo_shape)})
                        pseudo_embed = masked_embed['pseudo']
                        aux_embeddings[i * self.mask_batch_size: upper][idx_mask] = pseudo_embed.reshape((-1, aux_embeddings.shape[1]))
                    
                    counts = supp_counts + aux_counts
                    unnorm_prototypes = support_embeddings + aux_embeddings
                    counts = counts.type(torch.DoubleTensor).to(main_dev)
                    
                    norm_factor = torch.where(counts > eps, 1.0 / counts, 1.0)
                    norm_factor = norm_factor.unsqueeze(-1).expand(unnorm_prototypes.shape)
                    prototypes = unnorm_prototypes * norm_factor
                else:
                    supp_counts = supp_cls_matrix[:, 1].reshape(-1).type(torch.DoubleTensor).to(main_dev)
                    norm_factor = torch.where(supp_counts > eps, 1.0 / supp_counts, 1.0)
                    
                    # expand norms across row so that norm_factor.shape == support_embeddings.shape
                    norm_factor = norm_factor.unsqueeze(-1).expand(support_embeddings.shape)
                    norm_support_embeddings = support_embeddings * norm_factor 

                    prototypes = norm_support_embeddings
                
                torch.save({'protos': prototypes}, self.proto_file)
            
        accelerator.wait_for_everyone()

        # save prototypes to file so that all processes can sync
        protos = torch.load(self.proto_file)['protos']
        self.model.set_prototypes(protos)

    def _predict_epoch(self, data_loader, pred_classifier):
        # if we're testing, we need to overload the default dataset transform to use 84x84 images
        if self.dev_test:
            data_loader = copy.deepcopy(data_loader)
            data_loader.dataset.transform = self.transform_image()
        return super(PseudoShotTaglet, self)._predict_epoch(data_loader, pred_classifier)
