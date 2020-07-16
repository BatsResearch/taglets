from taglets.data.custom_dataset import CustomDataset
from torch.utils import data

from .module import Module
from ..pipeline import Taglet
from ..scads.interface.scads import Scads

import os
import re
import json
import random
import tempfile
import torch
import logging
import copy
import numpy as np
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision.models as models

from taglets.modules.zsl_kg_lite.class_encoders.transformer_model import TransformerConv
from taglets.modules.zsl_kg_lite.utils.core import save_model, l2_loss, set_seed, \
    convert_index_to_int, mask_l2_loss
from taglets.modules.zsl_kg_lite.imagenet_syns import IMAGENET_SYNS
from taglets.modules.zsl_kg_lite.example_encoders.resnet import ResNet

# graph related imports
from taglets.modules.zsl_kg_lite.utils.conceptnet import query_conceptnet
from taglets.modules.zsl_kg_lite.utils.graph import post_process_graph, \
    compute_union_graph, compute_embeddings, compute_mapping
from taglets.modules.zsl_kg_lite.utils.random_walk import graph_random_walk

log = logging.getLogger(__name__)


class ZeroShotModule(Module):
    def __init__(self, task):
        super().__init__(task)
        self.taglets = []


class ZeroShotTaglet(Taglet):
    def __init__(self, task):
        super().__init__(task)
        self.name = 'zero-shot'
        self.num_epochs = 1000
        self.save_dir = os.path.join('trained_model', self.name)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.save_path = os.path.join(self.save_dir, 'transformer.pt')

        self.options = {   
            "label_dim": 2049,
            "gnn": [
                {   "sample_nodes": True,
                    "relu": False,
                    "leaky_relu": False,
                    "num_sample": 50,
                    "dropout": True,
                    "output_dim": 2049,
                    "combine_self_concat": False,
                    "agg_self_loop":True,
                },
                {   
                    "sample_nodes": True,
                    "relu": False,
                    "leaky_relu": True,
                    "combine_self_concat": False,
                    "agg_self_loop":True,
                    "num_sample": 100,
                    "dropout": True,
                    "output_dim": 2048
                }
            ]
        }

        # using tempfile to create directories
        self.imagenet_graph_path = tempfile.mkdtemp()
        self.test_graph_path = tempfile.mkdtemp()

    def setup_imagenet_graph(self):
        root_path = Scads.get_root_path()
        Scads.open(self.task.scads_path)

        graph_path = "graph_data/imagenet_graph"
        glove_path = 'graph_data/'

        start_time = time.time()
        # TODO: get the imagenet graph
        query_conceptnet(graph_path,
                         'zsl_kg_lite/imagenet_syns.json',
                         '/data/bats/users/nnayak2/scads.sqlite3')

        # TODO: post process graph
        post_process_graph(graph_path)
        
        # TODO: take the union of the graph
        compute_union_graph(graph_path)

        # TODO: run random walk on the graph
        graph_random_walk(graph_path, k=20, n=10)

        end_time = time.time()

        # TODO: compute embeddings for the nodes
        compute_embeddings(graph_path, \
            '/data/bats/users/nnayak2/taglets/taglets/modules/graph_data/glove.840B.300d.txt')

        # TODO: compute mapping
        id_to_concept = json.load(open('zsl_kg_lite/id_to_concept.json'))
        compute_mapping(id_to_concept, graph_path)

        print('done!')

    def setup_fc(self):
        resnet = models.resnet50(pretrained=True)
        w = resnet.fc.weight
        b = resnet.fc.bias
        w.requires_grad = False
        b.requires_grad = False
        fc_vectors = torch.cat([w, b.unsqueeze(-1)], dim=1)
        fc_vectors = F.normalize(fc_vectors)

    return fc_vectors

    def setup_gnn(self, graph_path, device):
        # load graph
        adj_lists_path = os.path.join(graph_path, 'rw_adj_rel_lists.json')
        adj_lists = json.load(open(adj_lists_path))
        adj_lists = convert_index_to_int(adj_lists)

        # load embs
        concept_path = os.path.join(graph_path, 'concepts.pt')
        init_feats = torch.load(concept_path)

        model = TransformerConv(init_feats, adj_lists, device, self.options)

        return model

    def train(self, train_data_loader, val_data_loader, use_gpu):
        # 
        # TODO: ask from where to get the graph path, save path
        # the graph path for train and test will be different

        # TODO: check if this is right 
        if use_gpu:
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')
        
        log.debug('loading pretrained resnet50 fc weights')
        fc_vectors = self.setup_fc()
        fc_vectors.to(device)

        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001,
                            weight_decay=0.0005)

        self.model = self._train(fc_vectors, device)

    def _train(self, fc_vectors, device):
        
        log.debug("fc id to graph id mapping")
        mapping_path = os.path.join(self.train_graph_path, 'mapping.json')
        mapping = json.load(open(mapping_path))
        # 1000 because we are training on imagenet 1000
        imagenet_idx = torch.tensor([mapping[str(idx)] for idx in range(1000)]).to(device)

        log.debug('setting up gnn for traning')
        model = self.setup_gnn(self.train_graph_path, device)
        model.to(device)

        v_train, v_val = 0.95, 0.05
        n_trainval = len(fc_vectors)
        n_train = round(n_trainval * (v_train / (v_train + v_val)))
        log.info('num train: {}, num val: {}'.format(n_train, n_trainval - n_train))

        tlist = list(range(len(fc_vectors)))
        random.shuffle(tlist)

        trlog = {}
        trlog['train_loss'] = []
        trlog['val_loss'] = []
        trlog['min_loss'] = 0
        num_w = fc_vectors.shape[0]

        log.debug('zero-shot learning training started')
        for epoch in tqdm(range(1, 1000 + 1)):    
            model.train()
            for i, start in enumerate(range(0, n_train, 100)):
                end = min(start + 100, n_train)
                indices = tlist[start:end]
                output_vectors = model(imagenet_idx[indices])
                loss = l2_loss(output_vectors, fc_vectors[indices])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            model.eval()
            output_vectors = torch.empty(num_w, 2049, device=device)
            with torch.no_grad():
                for start in range(0, num_w, 100):
                    end = min(start + 100, num_w)
                    output_vectors[start: end] = model(imagenet_idx[start: end])

            train_loss = mask_l2_loss(output_vectors, fc_vectors, tlist[:n_train]).item()
            if v_val > 0:
                val_loss = mask_l2_loss(output_vectors, fc_vectors, tlist[n_train:]).item()
                loss = val_loss
            else:
                val_loss = 0
                loss = train_loss

            log.info('epoch {}, train_loss={:.4f}, val_loss={:.4f}'
                .format(epoch, train_loss, val_loss))

            # check if I need to save the model
            if trlog['val_loss']: 
                min_val_loss = min(trlog['val_loss'])
                if val_loss < min_val_loss:
                    save_model(model, self.save_path)
            else:
                save_model(model, self.save_path)

            trlog['train_loss'].append(train_loss)
            trlog['val_loss'].append(val_loss)

        # load the best model
        model.load_state_dict(torch.load(self.save_path))

        return model
    
    def _predict(self, resnet, class_rep, use_gpu):
        predictions = []
        with torch.no_grad():
            for data in loader:
                if use_gpu:
                    data = data.cuda()
                else:
                    data = data.cpu()

                feat = resnet(data) # (batch_size, d)
                feat = torch.cat([feat, torch.ones(len(feat)).view(-1, 1).cuda()], dim=1)

                logits = torch.matmul(feat, class_rep.t())

                pred = torch.argmax(logits, dim=1)
                predictions.extend([p.cpu().numpy().tolist() for p in pred])

        return predictions

    def execute(self, unlabeled_data_loader, use_gpu):
        if use_gpu:
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')
        
        self.model = self.setup_gnn(self.test_graph_path, device)

        log.debug('loading pretrained resnet')
        resnet = ResNet()
        resnet.eval()

        self.model.eval()
        log.debug('generating class representation')
        mapping_path = os.path.join(self.train_graph_path, 'mapping.json')
        mapping = json.load(open(mapping_path))
        conceptnet_idx = torch.tensor([mapping[str(idx)] for idx in len(mapping)]).to(device)
        class_rep = model(conceptnet_idx)

        # 
        log.debug('predicting')
        predictions = self._predict(resnet, class_rep, use_gpu)

        return predictions
