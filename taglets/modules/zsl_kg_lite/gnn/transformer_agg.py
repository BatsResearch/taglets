import torch
import torch.nn as nn

from allennlp.nn.util import masked_max, masked_mean, masked_softmax
from allennlp.modules.seq2seq_encoders.stacked_self_attention import StackedSelfAttentionEncoder

from taglets.modules.zsl_kg_lite.utils.core import pad_tensor, base_modified_neighbours

class TransformerAggregator(nn.Module):
    def __init__(self, features, input_dim, device, 
                 num_sample=50, sample_nodes=False, dropout=False,
                 num_heads=1, pd=None, hd=None, fh=None,
                 maxpool=False, dp=0.1, self_loop=False): 
        super(TransformerAggregator, self).__init__()

        self.features = features
        self.input_dim = input_dim
        self.device = device
        self.num_sample = num_sample
        self.sample_nodes = sample_nodes
        self.num_heads = num_heads
        self.proj_dim = pd or int(input_dim/2)
        self.hidden_dim = hd or input_dim
        self.ff_hidden = fh or int(input_dim/2)
        self.maxpool = maxpool
        self.self_loop = self_loop
        
        self.input_dim = input_dim
        if dropout:
            self.dropout = nn.Dropout(0.5)
        else:
            self.dropout = None

        self.attention = StackedSelfAttentionEncoder(input_dim,
                                                hidden_dim=self.hidden_dim, 
                                                projection_dim=self.proj_dim,
                                                feedforward_hidden_dim=self.ff_hidden,
                                                num_layers=1,
                                                num_attention_heads=self.num_heads,
                                                use_positional_encoding=False,
                                                attention_dropout_prob=dp)


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
                    samp_neighs[i].add(nodes[i])
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

        to_feats = torch.empty(len(samp_neighs), self.input_dim, device=self.device)

        modified_adj_nodes = base_modified_neighbours(samp_neighs, node_to_emb_idx)

        padded_tensor, mask = pad_tensor(modified_adj_nodes, mask=True)

        # sending padded tensor
        padded_tensor = padded_tensor.to(self.device)
        mask = mask.to(self.device)

        # embed matrix
        neigh_feats = embed_matrix[padded_tensor]

        attn_feats = self.attention(neigh_feats, mask)
        if self.maxpool:
            to_feats = masked_max(attn_feats, mask.unsqueeze(-1), dim=1)
        else:
            to_feats = masked_mean(attn_feats, mask.unsqueeze(-1), dim=1)

        return to_feats
