import os
import json
import click
import pandas as pd 
import random
from collections import Counter
from taglets.modules.zsl_kg_lite.utils import core

random.seed(0)

def random_walk(adj_lists, current_node, current_list, k):
    """The function computes the random walk over the graph

    Arguments:
        adj_lists {dict} -- dict containing lists of neighs
        current_node {int} -- the node id (usually indicates the starting point)
        current_list {list} -- the path walked so far
        k {int} -- the number of steps remaining

    Returns:
        list -- the list of transitions
    """
    if k == 0:
        return current_list
    else:
        node_rel = random.sample(adj_lists[current_node], 1)[0]
        # TODO: make it general purpose (should work even without the relation)
        if type(node_rel) == int:
            node = node_rel
        else:
            # tuple (node id, rel)
            node = node_rel[0]
        current_list.append(node)
        return random_walk(adj_lists, node, current_list, k-1)


def graph_random_walk(graph_path, k, n, seed=0):
    """The function is used to run random walk on the graph.

    Args:
        json_file (str): the path to the adj json 
        k (int): the length of the random walk
        n (int): the number of restarts
        seed (int, optional): seed value. Defaults to 0.
    """
    # set seed value
    random.seed(seed)

    print("creating loading adj lists")
    en_nodes_path = os.path.join(graph_path, 'en_nodes.csv')
    en_nodes = pd.read_csv(en_nodes_path)
    adj_rel_lists = json.load(open(os.path.join(graph_path)))
    
    rw_adj_lists = {}
    print("running random walks")
    rw_adj_lists = _run_random_walk(en_nodes, adj_rel_lists, k, n)

    # save the results
    print("saving the random walk results")
    with open(os.path.join(graph_path, 'rw_adj_rel_lists.json'), "w+") as fp:
        json.dump(rw_adj_lists, fp)

    print('done!')


def _run_random_walk(en_nodes, adj_rel_lists, k, n):
    rw_adj_lists = {}
    adj_rel_lists = core.convert_index_to_int(adj_rel_lists)

    for index, row in en_nodes.iterrows():
        transitions = []
        if index not in adj_rel_lists:
            continue

        if (index+1) % 10000 == 0:
            print(f'{index+1}/{len(en_nodes)}')

        for i in range(n):
            transitions += random_walk(adj_rel_lists, index, [], k)

        # filter 
        counts = Counter(transitions)
        nodes = set([neigh for neigh, rel in adj_rel_lists[index]])

        neigh_counts = dict([(neigh, count) for neigh, count in counts.items() if neigh in nodes])

        #
        for neigh, rel in adj_rel_lists[index]:
            if neigh not in neigh_counts:
                neigh_counts[neigh] = 0

        # add smoothing
        for neigh, rel in adj_rel_lists[index]:
            neigh_counts[neigh] += 1

        # hitting probability
        total = sum([count 
                     for neigh, count in neigh_counts.items()])

        hit_prob_dict = dict([(neigh, count/total * 1.0) 
                               for neigh, count in neigh_counts.items()])

        # save the probability
        rw_adj_lists[index] = [(neigh, rel, hit_prob_dict[neigh])
                               for neigh, rel in adj_rel_lists[index]]

    return rw_adj_lists
