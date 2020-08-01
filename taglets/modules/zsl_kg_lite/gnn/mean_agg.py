import random

import torch
import torch.nn as nn
import torch.nn.functional as F

class MeanAggregator(nn.Module):
    def __init__(self, features, device, num_sample=50, 
                 sample_nodes=False, dropout=False, 
                 self_loop=False): 
        super(MeanAggregator, self).__init__()

        self.features = features
        self.device = device
        self.num_sample = num_sample
        self.sample_nodes = sample_nodes
        self.self_loop = self_loop
        if dropout:
            self.dropout = nn.Dropout(0.5)
        else:
            self.dropout = None
        
    def forward(self, nodes, to_neighs):
        """Function computes the aggregated vector.

        Args:
            nodes (list): list of nodes
            to_neighs (list): list of list with node ids and relations

        Returns:
            torch.Tensor: tensors with aggregated vectors
        """

        _set = set
        if self.sample_nodes:         
            # sample neighs based on the hitting prob
            _neighs = [sorted(to_neigh, key=lambda x: x[2], reverse=True)[:self.num_sample] 
                       if len(to_neigh) >= self.num_sample else to_neigh for to_neigh in to_neighs]
            # 
            samp_neighs = []
            for i, adj_list in enumerate(_neighs):
                samp_neighs.append(set([node for node, rel, hp in adj_list]))
                if self.self_loop:
                    samp_neighs[i].add(nodes[i])
        else:
            # no sampling
            samp_neighs = to_neighs

        unique_nodes_list = sorted(list(set.union(*samp_neighs)))

        unique_nodes = {n:i for i,n in enumerate(unique_nodes_list)}
        mask = torch.zeros(len(samp_neighs), len(unique_nodes), device=self.device)
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]   
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        mask[row_indices, column_indices] = 1
        num_neigh = mask.sum(1, keepdim=True)
        mask = mask.div(num_neigh.clamp(1e-8))
        mask = mask.to(self.device)

        node_tensor = torch.tensor(unique_nodes_list, device=self.device)
        embed_matrix = self.features(node_tensor)

        if self.dropout is not None:
            embed_matrix = self.dropout(embed_matrix)

        to_feats = mask.mm(embed_matrix)

        return to_feats
