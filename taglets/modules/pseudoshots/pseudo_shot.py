import torch
import logging
import random
import os
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import copy 


log = logging.getLogger(__name__)

from ...data.custom_dataset import CustomImageDataset
from accelerate import Accelerator
from ..module import Module
from ...pipeline import Cache, ImageTaglet
from enum import IntEnum
from .masking_model import MaskingHead, multi_block_masking
from .utils import freeze_module, get_total_size
from ...scads import Scads, ScadsEmbedding

from .resnet12 import resnet12

accelerator = Accelerator()


class Encoder(IntEnum):
    RESNET_50 = 0

    # X1 = single width, X2 = double width (https://github.com/google-research/simclr)
    SIMCLR_50_X1 = 1
    SIMCLR_50_X2 = 2


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

        log.error(cls_prototypes)
        log.error('in forward')

        #log.error('in forward')
        if self.metric == Metric.COSINE:
            #log.error('nroamlizing protos')
            cls_prototypes = F.normalize(cls_prototypes, dim=-1)
            #log.error('setting x')
            #log.error(x)
            x = F.normalize(x, dim=-1)
            #log.error('done')

        #log.error('computing logits')
        logits = None
        
        x = x.reshape((x.shape[0], -1))
        if x.dim() == 2:
            if self.metric == Metric.DOT:
                logits = torch.mm(x, cls_prototypes.t())
            elif self.metric == Metric.COSINE:
                #log.error('comp!')
                #log.error(x.shape)
                #log.error(cls_prototypes.shape)
                a = F.normalize(x, dim=-1).type(torch.FloatTensor)
                b = F.normalize(cls_prototypes, dim=-1).t().type(torch.FloatTensor)
                #log.error(a.shape)
                #log.error(b.shape)
                #log.error(a.device)
                #log.error(b.device)
                #log.error(a.dtype)
                #log.error(b.dtype)
                logits = torch.matmul(a, b)
                #log.error('!!!')
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
        #log.error('returning')

        return logits.to(x.device)


class PseudoShotModule(Module):
    def __init__(self, task):
        super().__init__(task)
        ps_args = {}

        # few-shot params
        ps_args['n_shot'] = 1
        ps_args['n_way'] = 15
        ps_args['n_query'] = 15
        ps_args['n_pseudo'] = 15

        ps_args['img_encoder_type'] = Encoder.RESNET_50
        ps_args['img_encoder_ckpt_path'] = None
        ps_args['masking_ckpt_path'] = 'predefined/pseudoshots/masking_module.pt'
        ps_args['masking_args'] = {
            'channels': [640, 320, 1],
            'final_relu': False,
            'max_pool': False,
            'activation': 'sigmoid',
            'dropblock_size': 5
        }

        ps_args['metric'] = Metric.COSINE
        self.taglets = [PseudoShotTaglet(task, **ps_args)]


class PseudoShotTaglet(ImageTaglet):
    def __init__(self, task, masking_ckpt_path, masking_args, **kwargs):
        super().__init__(task)

        # only for development purposes
        self.dev_test = True
        self.dev_shape = (3, 84, 84)

        # training only parameters
        self.n_shot = kwargs.get('n_shot', 5)
        self.n_way = kwargs.get('n_way', 15)
        self.n_query = kwargs.get('n_query', 15)

        # PS budget
        self.n_pseudo = kwargs.get('n_pseudo', 100) if not os.environ.get("CI") else 1
        self.num_related_class = 5

        # mask batching for memory issues
        self.mask_batch_size = 20

        img_encoder_type = kwargs.get('img_encoder_type', Encoder.RESNET_50)
        self.img_encoder = self._set_img_encoder(img_encoder_type)
        self.masking_module = self._load_masking_module(masking_ckpt_path, **masking_args)
        self.masking_head = MaskingHead(self.masking_module)

        self.lr = kwargs.get('lr', self.lr)
        self._params_to_update = []

        #backbone = torch.nn.Sequential(*list(self.img_encoder.children())[:-1])
        #im_encoder_shape = self._get_model_output_shape(self.dev_shape, backbone)
        #log.info(im_encoder_shape)
        self.support_embeddings = torch.zeros((len(self.task.classes), 640*25))


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

    def _get_scads_data(self, label_map):
        data = Cache.get("pseudoshots", self.task.classes)
        if data is not None:
            image_paths, image_labels, all_related_class = data
        else:
            root_path = Scads.get_root_path()
            Scads.open(self.task.scads_path)
            ScadsEmbedding.load(self.task.scads_embedding_path)
            image_paths = []
            image_labels = []
            visited = set()

            def get_images(node, label, budget):
                if node.get_conceptnet_id() not in visited:
                    visited.add(node.get_conceptnet_id())
                    images = node.get_images_whitelist(self.task.whitelist)
                    sample_cnt = min(budget, len(images))
                    if sample_cnt == 0:
                        return False, budget

                    budget -= sample_cnt

                    images = random.sample(images, sample_cnt)
                    images = [os.path.join(root_path, image) for image in images]
                    image_paths.extend(images)
                    image_labels.extend([label] * len(images))
                    log.debug("Source class found: {}".format(node.get_conceptnet_id()))
                    return True, budget
                return False, budget

            all_related_class = 0
            aux_budget = self.n_pseudo
            k = 0
            log.error('task classes: ')
            log.error(self.task.classes)
            log.error('label_map: ')
            log.error(label_map)
            for i, conceptnet_id in enumerate(self.task.classes):
                cls_label = i
                cur_related_class = 0
                target_node = Scads.get_node_by_conceptnet_id(conceptnet_id)

                # prioritize data of the ame class
                valid, aux_budget = get_images(target_node, cls_label, aux_budget)
                if valid:
                    cur_related_class += 1
                    all_related_class += 1

                neighbors = ScadsEmbedding.get_related_nodes(target_node, self.n_pseudo)
                for neighbor in neighbors:
                    valid, aux_budget = get_images(neighbor, cls_label, aux_budget)
                    if valid:
                        cur_related_class += 1
                        all_related_class += 1
                        if cur_related_class >= self.num_related_class:
                            break

                if aux_budget > 0:
                    logging.warning('aux budget for %s is positive.' % conceptnet_id)

            log.error(f'misses: %d' % k)
            Scads.close()
            Cache.set('pseudoshots', self.task.classes,
                      (image_paths, image_labels, all_related_class))

        transform = self.transform_image()
        train_dataset = CustomImageDataset(image_paths,
                                           labels=image_labels,
                                           transform=transform)
        return train_dataset, all_related_class

    def _load_masking_module(self, ckpt_path, **kwargs):
        # TODO: determine how to sync encoder dim and masking module dim
        if self.dev_test:
            args = {'channels': [640, 320, 1],
                    'final_relu': False,
                    'max_pool': False,
                    'activation': 'sigmoid',
                    'dropblock_size': 5,
                    'inplanes': self.img_encoder.out_dim * 2}
            masking_module = multi_block_masking(**args)
            masking_module.load_state_dict(torch.load('predefined/pseudoshots/resnet12_mask.pth')['multi-block-masking_sd'])
            masking_module.eval()
            return masking_module

        masking_module = multi_block_masking(**kwargs)
        masking_module.load_state_dict(torch.load(ckpt_path))
        masking_module.eval()
        return masking_module

    def _set_img_encoder(self, img_encoder_type):
        if self.dev_test:
            encoder = resnet12(**{'avg_pool': False, 'drop_rate': 0.1, 'dropblock_size': 5})
            encoder.load_state_dict(torch.load('predefined/pseudoshots/resnet12.pth')['resnet12_sd'])
            encoder.eval()
            return encoder

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
        Note: train is somewhat of a misnomer because the nearest neighbor classifier is parameter-free.

        Steps:
        1. Iterate over each training example.
        2. Compute the encoding of each training example and save it to a list.
        3. For each class, compute
        """
        log.info('Beginning training')

        #train_data_loader = self._get_dataloader(data=train_data, shuffle=False)

        main_dev = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        if accelerator.is_local_main_process:
            train_data = copy.deepcopy(train_data)

            if self.dev_test:
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
                log.error('hi!!')
            log.error('yo')
            if supp_cls_matrix[:, 0].sum() != supp_cls_matrix.shape[0]:
                self.valid = False
                log.error('FSL ERROR: Number of classes in dataset not equal to number of task classes.')
                return

            log.error('supp_counts')
            # normalize summations
            supp_counts = supp_cls_matrix[:, 1].reshape(-1).type(torch.DoubleTensor).to(main_dev)
            log.error('norm_factor')
            log.error(supp_counts)
            eps = 0.5
            norm_factor = torch.where(supp_counts > eps, 1.0 / supp_counts, 1.0)
            # expand norms across row so that norm_factor.shape == support_embeddings.shape
            log.error('expand norm_factor')
            norm_factor = norm_factor.unsqueeze(-1).expand(support_embeddings.shape)
            log.error('norm_support_embeddings')
            log.error(norm_factor)
            log.error(support_embeddings)
            norm_support_embeddings = support_embeddings * norm_factor

            log.error('looking at label map')
            scads_train_data, all_related_class = self._get_scads_data(train_data.label_map)
            scads_data_loader = torch.utils.data.DataLoader(
                    dataset=scads_train_data, batch_size=self.batch_size, shuffle=False,
                    num_workers=0, pin_memory=True)

            log.error('scads grouping')
            # TODO: removing after debugging segfault
            all_related_class = 0
            if all_related_class > 0:
                aux_embeddings = torch.zeros(support_embeddings.shape, dtype=torch.int32).to(main_dev)
                aux_cls_matrix = torch.zeros(supp_cls_matrix.shape, dtype=torch.int32).to(main_dev)
                for x, y in scads_data_loader:
                    x, y = x.to(main_dev), y.to(main_dev)
                    x_embeds = img_encoder(x)

                    aux_cls_matrix, aux_embeddings = PseudoShotTaglet._group_encodings(x_embeds,
                                                                                           y,
                                                                                           aux_cls_matrix,
                                                                                           aux_embeddings)

                aux_counts = aux_cls_matrix[:, 1].reshape(-1)
                norm_factor = torch.where(aux_counts > 0, 1.0 / aux_counts.type(torch.DoubleTensor), 1.0)
                norm_factor = norm_factor.unsqueeze(-1).expand(aux_embeddings.shape)
                norm_aux_embeddings = aux_embeddings * norm_factor

                # generate masks
                joint_embeddings = torch.cat((norm_support_embeddings, norm_aux_embeddings), dim=1)

                n_batches = joint_embeddings.shape[0] // self.mask_batch_size
                assert joint_embeddings.shape[0] == aux_embeddings.shape[0]
                for i in range(max(n_batches, 1)):
                    upper = min((i + 1) * self.mask_batch_size, joint_embeddings.shape[0])
                    embed = joint_embeddings[i * self.mask_batch_size: upper]
                    pseudo = aux_embeddings[i * self.mask_batch_size: upper]

                    # only use masking module for classes with aux data
                    idx_mask = aux_counts[i * self.mask_batch_size: upper] > 0
                    masked_embed = masking_head({'embed': embed[idx_mask], 'pseudo': pseudo[idx_mask]})['pseudo']
                    aux_embeddings[idx_mask] = masked_embed.reshape((pseudo.shape[0], -1))

                counts = supp_counts + aux_counts
                unnorm_prototypes = support_embeddings + aux_embeddings
                norm_factor = torch.where(counts > 0, 1.0 / counts.type(torch.DoubleTensor), 1.0)
                norm_factor = norm_factor.unsqueeze(-1).expand(unnorm_prototypes.shape)
                prototypes = unnorm_prototypes * norm_factor
            else:
                prototypes = norm_support_embeddings
            
            log.error('setting protos')
            #self.model.set_prototypes(prototypes)
            torch.save({'protos': prototypes}, self.proto_file)
            
        accelerator.wait_for_everyone()
        protos = torch.load(self.proto_file)['protos']
        self.model.set_prototypes(protos)

    def _get_dataloader(self, data, shuffle, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        return accelerator.prepare(torch.utils.data.DataLoader(
            dataset=data, batch_size=batch_size, shuffle=shuffle,
            num_workers=0, pin_memory=True
        ))

    def _predict_epoch(self, data_loader, pred_classifier):
        outputs = []
        labels = []
        
        data_loader.dataset.transform = self.transform_image()
        for batch in data_loader:
            if isinstance(batch, list):
                inputs, targets = batch
            else:
                inputs, targets = batch, None
            
            with torch.set_grad_enabled(False):
                output = pred_classifier(inputs)
                log.error('out')
                log.error(output.device)
                log.error('a')
                a = output.detach()
                log.error('b')

                b = accelerator.gather(a)
                log.error('c')
                c = b.cpu()
                log.error('d')
                d = F.softmax(c, 1)
                log.error('append')
                outputs.append(d)
                
                log.error('targets')

                if targets is not None:
                    labels.append(accelerator.gather(targets.detach()).cpu())
        
        log.error('to-numpy')

        outputs = torch.cat(outputs).numpy()
        if len(labels) > 0:
            labels = torch.cat(labels).numpy()
            
        # Accelerate pads the dataset if its length is not divisible by the "actual" batch size
        # so we need to remove the extra elements

        log.error(outputs)
        dataset_len = len(data_loader.dataset)
        log.error(len(data_loader.dataset))
        outputs = outputs[:dataset_len]
        labels = labels[:dataset_len]
        log.error('pred-ret')
        log.error(outputs)
        log.error(labels)
        return outputs, labels

    def _train(self):
        # Pretrain Masking Module
        pass
