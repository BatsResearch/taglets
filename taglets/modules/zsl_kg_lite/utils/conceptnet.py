"""querying the graph
"""
import os
import click
import sqlite3
import json
import pandas as pd
from tqdm import tqdm
import itertools
import time
import copy
import shutil

TABLE_COLUMNS = {
    'nodes': ['id', 'uri'],
    'sources': ['id', 'uri'],
    'relations': ['id', 'uri', 'directed'],
    'edges': ['id', 'uri', 'relation_id', 'start_id', 'end_id', 'weight', 'data'],
    'edges_gin': ['edge_id', 'weight', 'data'],
    'edge_features': ['rel_id', 'direction', 'node_id', 'edge_id']
}

def run_commands(connection, commands):
    cursor = connection.cursor()
    for cmd in commands:
        cursor.execute(cmd)
    connection.commit()


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def query_conceptnet(output_path, conceptnet_path, database_path, n=2):
    """The function is used to query the conceptnet database. 

    Args:
        output_path (str): the path to the output directory

        conceptnet_path (str): json file containing concept and its synonyms;
        {
            "/c/en/horse": ["/c/en/house/n"],
            ...
        }
        database_path (str): conceptnet db path
        n (int, optional): the maximum number of hops from a concept. Defaults to 2.
    """
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
    # conceptnet path

    import json
    with open(conceptnet_path) as fp:
        concept_syn = json.load(fp)
    
    concepts = []
    for concept, syns in concept_syn.items():
        concepts.append(concept)
        concepts += syns
    # 
    idx_to_relation = {}
    directed_dict = {"t": True, "f": False}
    relations = cursor.execute("SELECT id, type, is_directed from relations").fetchall()
    
    # get all the concepts related to the type
    idx_to_concept = {}
    label_node_ids = set()
    syn_id_to_node_id = {}
    for concept, syns in concept_syn.items():
        # concept
        x = cursor.execute("SELECT * from nodes where conceptnet_id=\""+concept+"\" LIMIT 1").fetchall()
        
        if not x:
            raise Exception(concept + " not found")
        
        label_node_id, node_uri = x[0]
        label_node_ids.add(label_node_id)

        for _syn in syns:
            x = cursor.execute("SELECT * from nodes where conceptnet_id=\""+_syn+"\" LIMIT 1").fetchall()

            for node_id, node_uri in x:
                syn_id_to_node_id[node_id] = label_node_id
                label_node_ids.add(node_id)

    hops = []
    hops.append(list(label_node_ids))
    adj_list = []

    for i in range(n):
        print("Hop ", i)
        times = []
        new_nodes = set()
        for batch_nodes in chunks(hops[i], 5000):
            for x in ['start_node', 'end_node']:
                query_string = "select start_node, end_node, relation_type, weight from edges where"        
                for node_id in batch_nodes:
                    query_string += " " + x + "=" + str(node_id)
                    query_string += " or"

                query_string = query_string[: -3]
                neigh_concepts = cursor.execute(query_string).fetchall()
                adj_list.extend(neigh_concepts)
                all_concepts = set((itertools.chain.from_iterable(hops)))
                for start_id, end_id, relation_id, weight in neigh_concepts:
                    if start_id not in all_concepts:
                        new_nodes.add(start_id)
                    if end_id not in all_concepts:
                        new_nodes.add(end_id)
            
        hops.append(list(new_nodes))
    
    # get all concepts
    all_concepts = set((itertools.chain.from_iterable(hops)))
    nodes = []
    for c in all_concepts:
        x = cursor.execute("SELECT * from nodes where conceptnet_id=\""+str(c)+"\" LIMIT 1").fetchall()
        for node_id, node_uri in x:
            nodes.append((node_id, node_uri))

    # create directory if not present
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # copy conceptnet path to output dir and save as syns.json
    syns_path = os.path.join(output_path, 'syns.json')
    shutil.copy(conceptnet_path, syns_path)

    # save the id relation directed
    rel_file = os.path.join(output_path, 'relations.csv')
    df = pd.DataFrame(relations, columns=['id', 'uri', 'directed'])
    df.to_csv(rel_file, index=False)

    # save the id to node_uri mapping
    node_file = os.path.join(output_path, 'nodes.csv')
    df = pd.DataFrame(nodes, columns=['id', 'uri'])
    df.to_csv(node_file, index=False)

    # save start_id, end_id, relation_id, weight
    edge_file = os.path.join(output_path, 'unreplaced_edges.csv')
    df = pd.DataFrame(adj_list, columns=['start_id', 'end_id', 'relation_id', 'weight'])
    df.to_csv(edge_file, index=False)

    # save start_id, end_id, relation_id, weight
    edge_file = os.path.join(output_path, 'edges.csv')

    # replace adj list with the syn ids
    replaced_edges = []
    for start_id, end_id, relation_id, weight in adj_list:
        
        if start_id in syn_id_to_node_id:
            start_id = syn_id_to_node_id[start_id]
        
        if end_id in syn_id_to_node_id:
            end_id = syn_id_to_node_id[end_id]
    
        replaced_edges.append((start_id, end_id, relation_id, weight))

    edge_file = os.path.join(output_path, 'edges.csv')
    df = pd.DataFrame(replaced_edges, columns=['start_id', 'end_id', 'relation_id', 'weight'])
    df.to_csv(edge_file, index=False)

    print('done!')
