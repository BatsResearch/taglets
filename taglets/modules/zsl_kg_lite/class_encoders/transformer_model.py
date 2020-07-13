
import torch
import torch.nn as nn

from zsl_kg_lite.gnn.transformer_agg import TransformerAggregator
from zsl_kg_lite.gnn.combine import Combine

class TransformerConv(nn.Module):
    def __init__(self, features, adj_lists, device, options):
        """The Transformer class encoder for generating class representations.
        The transformer class encoder uses transformer aggregator and combine
        function to generate the class representaitons. 

        Args:
            features (nn.Tensor): tensor contains the initial node embeddings. 
            adj_lists (dict): the graph with nodes and edges.
            device (nn.device): the device information 
            options (dict): this is a dictionary that contains 
                            additional information for the gnn.
                            Below is a sample. 
                            Note: the gnn module is ordered according 
                            to hops.

                            {   "label_dim": 2049,
                                "gnn": [
                                    {   "sample_nodes": True,
                                        "relu": False,
                                        "leaky_relu": True,
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
        """
        super(TransformerConv, self).__init__()

        self.device = device
        self.adj_lists = adj_lists
        self.options = options
        self.init_feat =  nn.Embedding.from_pretrained(features, freeze=True)

        self.label_dim = options['label_dim']

        # TODO: give option have complex properties
        self.gnn_modules = []
        input_dim = features.size(1)
        current_enc = self.init_feat
        for i in options['gnn'][::-1]:
            agg = TransformerAggregator(current_enc, input_dim,
                                        device, num_sample=i['num_sample'],
                                        sample_nodes=i['sample_nodes'],
                                        dropout=i['dropout'], 
                                        self_loop=i['agg_self_loop'])

            enc = Combine(current_enc, input_dim, i['output_dim'],
                          adj_lists=adj_lists, aggregator=agg, device=device,
                          self_concat=i['combine_self_concat'], 
                          leaky_relu=i['leaky_relu'],
                          relu=i['relu'])

            self.gnn_modules.append(enc)
            current_enc = enc
            input_dim = i['output_dim']

        self.add_module('conv', self.gnn_modules[-1])

    def forward(self, label_idx):
        return self.gnn_modules[-1](label_idx)
