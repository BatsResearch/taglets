
import os
import re
import random

import torch
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import pandas as pd


def pad_tensor(adj_nodes_list, mask=False):
    """Function pads the neighbourhood nodes before passing through the 
    aggregator.

    Args:
        adj_nodes_list (list): the list of node neighbours
        mask (bool, optional): if true, create a tensor with 1s and 0s for masking. Defaults to False.

    Returns:
        tuple: one of two tensors containing the padded tensor and mask (if true)
    """
    max_len = max([len(adj_nodes) for adj_nodes in adj_nodes_list])

    padded_nodes = []
    _mask = []
    for adj_nodes in adj_nodes_list:
        x = list(adj_nodes)
        
        x += [0] * (max_len - len(adj_nodes))
        padded_nodes.append(x)
        _mask.append([1] * len(adj_nodes) + [0] * (max_len - len(adj_nodes)))

    if not mask:
        return torch.tensor(padded_nodes)

    # returning the mask as well
    return torch.tensor(padded_nodes), torch.tensor(_mask)


def base_modified_neighbours(adj_nodes_list, idx_mapping):
    """function maps the node indices to new indices using a mapping
    dictionary.

    Args:
        adj_nodes_list (list): list of list containing the node ids
        idx_mapping (dict): node id to mapped node id

    Returns:
        list: list of list containing the new mapped node ids
    """
    new_adj_nodes_list = []
    for adj_nodes in adj_nodes_list:
        new_adj_nodes = []
        for node in adj_nodes:
            new_adj_nodes.append(idx_mapping[node])        
        new_adj_nodes_list.append(new_adj_nodes)
    
    return new_adj_nodes_list


def get_rel_ids(adj_lists, neigh_sizes, node_ids):
    """Function to get all the rel ids

    Arguments:
        adj_lists {dict} -- dictionary containing list of list
        neigh_sizes {list} -- list containing the sample size of the neighbours
        node_ids {list} -- contains the initial train ids

    Returns:
        set -- returns the set of relations that are part of the training
    """
    all_rels = []
    nodes = node_ids
    for sample_size in neigh_sizes:
        to_neighs = [adj_lists[node] for node in nodes]
        _neighs = [sorted(to_neigh, key=lambda x: x[2], reverse=True)[:sample_size] 
                        if len(to_neigh) >= sample_size else to_neigh for to_neigh in to_neighs]
        _node_rel = []
        # nodes = []
        for neigh in _neighs:
            for node, rel, hp in neigh:
                all_rels.append(rel)
                nodes.append(node)  
  
    all_rels = set(all_rels)
    return all_rels


def prune_graph(adj_lists, relations):
    """The function is used to prune graph based on the relations 
    that are present in the training

    Arguments:
        adj_lists {dict} -- dictionary containing the graph
        relations {set} -- list of relation ids

    Returns:
        dict -- pruned graph
    """
    pruned_adj_list = {}
    for node, adj in adj_lists.items():
        pruned_adj_list[node] = []
        for neigh_node, rel, hp in adj:
            if rel in relations:
                pruned_adj_list[node].append((neigh_node, rel, hp))
    
    return pruned_adj_list


def convert_index_to_int(adj_lists):
    """Function to convert the node indices to int
    """
    new_adj_lists = {}
    for node, neigh in adj_lists.items():
        new_adj_lists[int(node)] = neigh
    
    return new_adj_lists


def normt_spm(mx, method='in'):
    if method == 'in':
        mx = mx.transpose()
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

    if method == 'sym':
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -0.5).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = mx.dot(r_mat_inv).transpose().dot(r_mat_inv)
        return mx


def spm_to_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack(
            (sparse_mx.row, sparse_mx.col))).long()
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def create_dirs(path):
    """create directories if path doesn't exist

    Arguments:
        path {str} -- path of the directory
    """
    os.makedirs(path, exist_ok=True)


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # if you are suing GPU
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# TODO: move to a better place
def mask_l2_loss(a, b, mask):
    return l2_loss(a[mask], b[mask])


def l2_loss(a, b):
    return ((a - b)**2).sum() / (len(a) * 2)


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        cuda_device = 0
    else:
        device = torch.device('cpu')
        cuda_device = -1
    return device, cuda_device


def save_model(model, save_path):
    """The function is used to save the model

    Arguments:
        model {nn.Model} -- the model
        save_path {str} -- model save path
    """
    # TODO: test this module
    torch.save(model.state_dict(), save_path)
