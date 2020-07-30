
import torch
import torch.nn as nn
from torch.nn import init

from zsl_kg_lite.gnn.attention_agg import AttnAggregator
from zsl_kg_lite.gnn.combine import AttnCombine

class GATConv(nn.Module):
    def __init__(self, features, adj_lists, device, options=None):
        super(GATConv, self).__init__()
        
        # vectors = vectors
        self.adj_lists = adj_lists
        self.device = device
        self.init_feat =  nn.Embedding.from_pretrained(features, freeze=True)

        self.label_dim = options['label_dim']

        self.gnn_modules = []
        input_dim = features.size(1)
        current_enc = self.init_feat
        for i in options['gnn'][::-1]:
            agg = AttnAggregator(current_enc, input_dim, i['output_dim'], 
                                 device, num_sample=i['num_sample'],
                                 sample_nodes=i['sample_nodes'],
                                 dropout=i['dropout'],
                                 self_loop=i['agg_self_loop'])

            enc = AttnCombine(current_enc, adj_lists=adj_lists, 
                              aggregator=agg, device=device,
                              leaky_relu=i['leaky_relu'],
                              relu=i['relu'])
            
            self.gnn_modules.append(enc)
            current_enc = enc
            input_dim = i['output_dim']

        self.add_module('conv', self.gnn_modules[-1])

    def forward(self, label_idx):
        return self.gnn_modules[-1](label_idx)