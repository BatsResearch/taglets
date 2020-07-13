import torch
import torch.nn as nn
from torch.nn import init

from allennlp.nn.util import masked_max, masked_mean, masked_softmax

from zsl_kg_lite.utils.core import pad_tensor, base_modified_neighbours


class AttnAggregator(nn.Module):
    def __init__(self, features, input_dim, output_dim, device, 
                 num_sample=50, sample_nodes=False, dropout=False,
                 self_loop=False): 
        """
        GAT: Attention Aggregator
        """
        super(AttnAggregator, self).__init__()

        self.features = features
        self.device = device
        self.num_sample = num_sample
        self.sample_nodes = sample_nodes
        
        self.input_dim = input_dim
        if dropout:
            self.dropout = nn.Dropout(0.5)
        else:
            self.dropout = None

        self.proj = nn.Linear(input_dim, output_dim, bias=False)

        init.xavier_uniform_(self.proj.weight)

        self.attn_src = nn.Linear(output_dim, 1, bias=False)
        self.attn_dst = nn.Linear(output_dim, 1, bias=False)

        self.leaky_relu = nn.LeakyReLU(0.2)

        self.self_loop = self_loop

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
            # change ds
            samp_neighs = []
            for i, adj_list in enumerate(_neighs):
                samp_neighs.append(set([node for node, rel, hp in adj_list]))
                if self.self_loop:
                    samp_neighs[i].add(int(nodes[i]))
        else:
            # no sampling
            samp_neighs = to_neighs

        unique_nodes_list = list(set.union(*samp_neighs))

        # get the unique nodes
        unique_nodes = list(set(unique_nodes_list))
        node_to_emb_idx = {n:i for i,n in enumerate(unique_nodes)}
        unique_nodes_tensor = torch.tensor(unique_nodes, device=self.device)

        embed_matrix = self.features(unique_nodes_tensor)
        if self.dropout is not None:
            embed_matrix = self.dropout(embed_matrix)
        
        # get new features
        embed_matrix_prime = self.proj(embed_matrix) 

        to_feats = torch.empty(len(samp_neighs), self.input_dim, device=self.device)
        modified_adj_nodes = base_modified_neighbours(samp_neighs, node_to_emb_idx)

        # 
        padded_tensor, mask = pad_tensor(modified_adj_nodes, mask=True)
        # sending padded tensor
        padded_tensor = padded_tensor.to(self.device)
        mask = mask.to(self.device)

        dst_nodes = []
        max_length = mask.size(1)
        for _node in nodes:
            dst_nodes.append([node_to_emb_idx[_node]] * max_length)

        dst_tensor = torch.tensor(dst_nodes).to(self.device)

        # embed matrix
        neigh_feats = embed_matrix_prime[padded_tensor]
        dst_feats = embed_matrix_prime[dst_tensor]

        # attention 
        dst_attn = self.leaky_relu(self.attn_dst(dst_feats))
        neigh_attn = self.leaky_relu(self.attn_src(neigh_feats))

        edge_attn = dst_attn + neigh_attn

        attn = masked_softmax(edge_attn, mask.unsqueeze(-1), dim=1)

        # multiply attention
        to_feats = torch.sum(attn * neigh_feats, dim=1)

        return to_feats
