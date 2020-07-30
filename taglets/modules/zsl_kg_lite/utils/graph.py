"""
post process
compute mapping
compute embeddings
union graph
"""
import os
import re
import json
import itertools
import pandas as pd
import numpy as np

import torch
import torch.nn.functional as F
from .conceptnet import chunks


def post_process_graph(graph_path):
    """The function post processes the graph after extraction. 
    The post processing involves removing non-english nodes in 
    graph, mapping the edge ids to node ids.

    Args:
        graph_path (str): the directory path of the graph
    """
    # read the graph path
    edges_path = os.path.join(graph_path, 'unreplaced_edges.csv')
    unreplaced_edges = pd.read_csv(edges_path)
    unreplaced_edges.drop_duplicates()

    # read all the nodes
    nodes_path = os.path.join(graph_path, 'nodes.csv')
    nodes = pd.read_csv(nodes_path)

    # relations
    rel_path = os.path.join(graph_path, 'relations.csv')
    relations = pd.read_csv(rel_path)

    # all only english nodes
    en_nodes = nodes.uri.str.contains(r'\/c\/en\/')
    en_nodes = nodes[en_nodes]

    # filter edges
    print("filtering en edges")
    start_edges = unreplaced_edges['start_id'].isin(en_nodes['id'])
    end_edges = unreplaced_edges['end_id'].isin(en_nodes['id'])
    en_edges_truth = end_edges & start_edges
    en_edges = unreplaced_edges[en_edges_truth]

    # start id to end id (with new nodes)
    print("mapping edges to new node ids")
    node_id_to_idx = dict([(node_id, idx) for idx, node_id in enumerate(en_nodes['id'])])
    mapped_edges = []
    for index, row in en_edges.iterrows():
        start_id = node_id_to_idx[row['start_id']]
        end_id = node_id_to_idx[row['end_id']]
        mapped_edges.append((start_id, end_id, int(row['relation_id']), row['weight']))

    print("saving en nodes ...")
    en_nodes_path = os.path.join(graph_path, 'en_nodes.csv')
    en_nodes.to_csv(en_nodes_path, index=False)
    
    print("saving mapped edges ...")
    mapped_edge_file = os.path.join(graph_path, 'en_mapped_edges.csv')
    mapped_edges = pd.DataFrame(mapped_edges, columns=['start_id', 'end_id', 'relation_id', 'weight'])
    mapped_edges.to_csv(mapped_edge_file, index=False)
    
    print('done!')


def compute_union_graph(graph_path):
    """The function is used to compute the union graph. The union
    graph is the union of the concept with the nodes of its 
    synonyms.

    Args:
        graph_path (str): the directory path of the graph
    """

    print("loading the en nodes")
    en_nodes_path = os.path.join(graph_path, 'en_nodes.csv')
    en_nodes = pd.read_csv(en_nodes_path)

    idx_to_node = dict(en_nodes['uri'])
    node_to_idx = dict([(node, idx) for idx, node in idx_to_node.items()])

    # 
    print("load syns for the nodes")
    syns_path = os.path.join(graph_path, 'syns.json')
    syns_dict = json.load(open(syns_path))

    # syns id
    syns_id = {}
    for key, syns in syns_dict.items():
        if key not in node_to_idx:
            print(key)
            continue

        replaced_id = node_to_idx[key]

        for syn in syns:
            if syn in node_to_idx:
                syns_id[node_to_idx[syn]] = replaced_id
    
    # replace all the syn edges in key edges 
    en_edges = pd.read_csv(os.path.join(graph_path, "en_mapped_edges.csv"))

    only_en_edges = en_edges[['start_id', 'end_id']]

    union_edges = only_en_edges.replace(list(syns_id.keys()), list(syns_id.values()))
    union_edges[['relation_id', 'weight']] = en_edges[['relation_id', 'weight']]

    # remove self loop
    print("saving the union edges")
    union_edge_file = os.path.join(graph_path, 'union_en_edges.csv')
    union_edges.to_csv(union_edge_file, index=False)

    # saving the union adj list
    print("computing the adj lists")
    union_adj_lists = {}
    union_adj_rel_lists = {}

    opp_union_edges = union_edges.copy()
    opp_union_edges[['start_id', 'end_id']] = union_edges[['end_id', 'start_id']]
    concat_union_edges = pd.concat((union_edges, opp_union_edges), ignore_index=True)

    union_adj_rel_lists = concat_union_edges[['start_id', 
                                              'end_id',
                                              'relation_id']].set_index('start_id').apply(tuple, 1)\
             .groupby(level=0).agg(lambda x: set(x.values))\
             .to_dict()
    union_adj_lists = concat_union_edges[['start_id',
                                          'end_id']].set_index('start_id').apply(tuple, 1)\
            .groupby(level=0).agg(lambda x: set(x.values))\
             .to_dict()
    
    new_adj_lists = {}
    for node, adj in union_adj_lists.items():
        new_adj_lists[node] = list(itertools.chain.from_iterable(adj))

    print("saving union adj lists")
    with open(os.path.join(graph_path, 'union_adj_lists.json'), 'w+') as fp:
        json.dump(new_adj_lists, fp)

    new_adj_lists = {}
    for node, adj in union_adj_rel_lists.items():
        new_adj_lists[node] = list(adj)
    
    with open(os.path.join(graph_path, 'union_adj_rel_lists.json'), 'w+') as fp:
        json.dump(new_adj_lists, fp)

    print('done!')


def compute_mapping(id_to_concept, graph_path):

    # load the en_nodes
    print('loading the en nodes')
    en_nodes = pd.read_csv(os.path.join(graph_path, 'en_nodes.csv'))

    idx_to_node_uri = en_nodes['uri'].to_dict()
    node_uri_to_idx = dict([(node_id, idx) for idx, node_id in idx_to_node_uri.items()])

    print('creating mapping json file')
    mapping = {}
    for _id, concept in id_to_concept.items():
        mapping[_id] = node_uri_to_idx[concept]
    
    print('saving id to concept')
    with open(os.path.join(graph_path, 'id_to_concept.json'), 'w+') as fp:
        json.dump(id_to_concept, fp)
    
    print('saving mapping')
    with open(os.path.join(graph_path, 'mapping.json'), 'w+') as fp:
        json.dump(mapping, fp)
    
    print('done!')


def compute_embeddings(graph_path, glove_path):
    """The function is used to compute the initial node embeddings
    for all the nodes in the graph. 

    Args:
        graph_path (str): path to the conceptnet subgraph directory
        glove_path (str): path to the glove file
    """
    # load the english nodes
    print("loading en nodes")
    en_nodes_path = os.path.join(graph_path, 'en_nodes.csv')
    en_nodes = pd.read_csv(en_nodes_path)

    # get words from the nodes
    print("extract individual words from concepts")
    words = set()
    all_concepts = []
    for index, node in en_nodes.iterrows():
        concept_words = get_individual_words(node['uri'])
        all_concepts.append(concept_words)
        for w in concept_words:
            words.add(w)
    
    # 
    word_to_idx = dict([(word, idx+1) for idx, word in enumerate(words)])
    word_to_idx["<PAD>"] = 0
    idx_to_word = dict([(idx, word) for word, idx in word_to_idx.items()])

    # load glove 840
    print("loading glove from file")
    glove = load_embeddings(glove_path)
    
    # get the word embedding
    embedding_matrix = torch.zeros(len(word_to_idx), 300)
    for idx, word in idx_to_word.items():
        if word in glove:
            embedding_matrix[idx] = torch.Tensor(glove[word])
    
    # 
    print("padding concepts")
    max_length = max([len(concept_words) for concept_words in all_concepts])
    padded_concepts = []
    for concept_words in all_concepts:
        concept_idx = [word_to_idx[word] for word in concept_words]
        concept_idx += [0] * (max_length - len(concept_idx))
        padded_concepts.append(concept_idx)
    
    # add the word embeddings of indivual words
    print("adding the word embeddings and l2 norm-> conceptnet embeddings")
    concept_embs = torch.zeros((0, 300))
    padded_concepts = torch.tensor(padded_concepts)
    for pc in chunks(padded_concepts, 100000):
        concept_words = embedding_matrix[pc]
        embs = torch.sum(concept_words, dim=1)
        embs = F.normalize(embs)
        concept_embs = torch.cat((concept_embs, embs), dim=0)

    # save the conceptnet embs
    print('saving the concept embeddings')
    concept_path = os.path.join(graph_path, 'concepts.pt')
    torch.save(concept_embs, concept_path)

    print('done!')


def load_embeddings(file_path):
    """file to load glove
    """
    embeddings = {}
    with open(file_path) as fp:
        for line in fp:
            fields = line.rstrip().split(' ')
            vector = np.asarray(fields[1:], dtype='float32')
            embeddings[fields[0]] = vector

    return embeddings


def get_individual_words(concept):
    """extracts the individual words from a concept
    """
    clean_concepts = re.sub(r"\/c\/[a-z]{2}\/|\/.*", "", concept)
    return clean_concepts.strip().split("_")
