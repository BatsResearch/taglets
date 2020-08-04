import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

class Combine(nn.Module):
    """
    Encodes a node's using 'convolutional' GraphSage approach
    """
    def __init__(self, features, input_dim, 
            output_dim, adj_lists, aggregator, device,
            self_concat=False,
            relu=False, leaky_relu=False): 
        super(Combine, self).__init__()

        self.features = features
        self.input_dim = input_dim
        self.adj_lists = adj_lists
        self.aggregator = aggregator

        self.self_concat = self_concat
        self.output_dim = output_dim
        self.device = device

        if not self.self_concat:
            self.w = nn.Parameter(
                    torch.empty(input_dim, output_dim).to(self.device))
            self.b = nn.Parameter(torch.zeros(output_dim).to(self.device))
        else:
            self.w = nn.Parameter(
                    torch.empty(2 * input_dim, output_dim).to(self.device))
            self.b = nn.Parameter(torch.zeros(output_dim).to(self.device))

        init.xavier_uniform_(self.w)

        # 
        if relu:
            self.relu = nn.ReLU()
        else:
            self.relu = None
        
        # TODO: give option to change to change this 
        # giving preference for relu over leaky relu
        if leaky_relu and not relu:
            self.relu = nn.LeakyReLU(0.2)

    def forward(self, nodes):
        """
        Generates embeddings for a batch of nodes.

        nodes     -- list of nodes
        """
        nodes_list = [int(node) for node in nodes]
        neigh_feats = self.aggregator.forward(nodes_list, \
                                              [self.adj_lists[node] \
                                              for node in nodes_list])
        if self.self_concat:
            if type(nodes) == list:
                nodes = torch.tensor(nodes).to(self.device)
            self_feats = self.features(nodes)
            combined = torch.cat([self_feats, neigh_feats], dim=1)
        else:
            combined = neigh_feats

        output = torch.mm(combined, self.w) + self.b
        
        if self.relu is not None:
            output = self.relu(output)

        output = F.normalize(output)

        return output


class AttnCombine(nn.Module):
    """
    Encodes a node's using 'convolutional' GraphSage approach
    """
    def __init__(self, features, adj_lists, aggregator,
                 device, relu=False, leaky_relu=False):
        super(AttnCombine, self).__init__()

        # TODO: give an option to use self concat

        self.features = features
        self.adj_lists = adj_lists
        self.aggregator = aggregator

        self.device = device
        # 
        if relu:
            self.relu = nn.ReLU()
        else:
            self.relu = None
        
        # giving preference for relu over leaky relu
        if leaky_relu and not relu:
            self.relu = nn.LeakyReLU(0.2)

    def forward(self, nodes):
        """
        Generates embeddings for a batch of nodes.

        nodes     -- list of nodes
        """
        nodes_list = [int(node) for node in nodes]
        output = self.aggregator.forward(nodes_list, [self.adj_lists[node] for node in nodes_list])


        if self.relu is not None:
            output = self.relu(output)

        return F.normalize(output)
