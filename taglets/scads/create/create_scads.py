import os
from .scads_classes import Node, Edge, Relation, Base, engine, Session


def get_relations():
    """
    Get all Relations in conceptnet.
    :return: A list of Relations
    """
    cwd = os.getcwd()
    all_relations = []
    with open(os.path.join(cwd, 'sql_data', 'relations.csv'), encoding="utf8") as relation_file:
        for line in relation_file:
            if line.strip():
                r_key, r_name, r_type = line.strip().split()
                relation = Relation(key=r_key, name=r_name, type=r_type)
                all_relations.append(relation)
    return all_relations


def get_nodes():
    """
    Get all Nodes in conceptnet.
    :return: A list of Nodes
    """
    cwd = os.getcwd()
    all_nodes = []
    with open(os.path.join(cwd, 'sql_data', 'nodes.csv'), encoding="utf8") as node_file:
        for line in node_file:
            if line.strip():
                n_key, n_name = line.strip().split()
                node = Node(key=n_key, name=n_name)
                all_nodes.append(node)
    return all_nodes


def get_edges():
    """
    Get all edges in conceptnet.
    :return: A list of Edges
    """
    cwd = os.getcwd()
    all_edges = []
    with open(os.path.join(cwd, 'sql_data', 'edges.csv'), encoding="utf8") as edge_file:
        for line in edge_file:
            if line.strip():
                key, uri, relation_key, start_node_key, end_node_key, weight = line.strip().split()[:6]
                info = " " .join(line.strip().split()[6:])
                edge = Edge(key=key,
                            URI=str(uri),
                            relation_key=relation_key,
                            start_node_key=start_node_key,
                            end_node_key=end_node_key,
                            weight=weight,
                            info=info)
                all_edges.append(edge)
    return all_edges


def add_conceptnet():
    """
    Add Nodes, Edges, and Relations from conceptnet into the database.
    :return: None
    """
    # Generate database schema
    Base.metadata.create_all(engine)
    session = Session()

    # Read nodes, edges, and relations from files
    all_relations = get_relations()
    all_nodes = get_nodes()
    all_edges = get_edges()

    # Add relationships to the database
    # Assumes the relations in all_relations are already sorted based on their keys and the keys start at 0
    for edge in all_edges:
        edge.relation = all_relations[int(edge.relation_key)]
        edge.end_node = all_nodes[int(edge.end_node_key)]

    # Insert data into tables
    session.add_all(all_relations)
    session.add_all(all_nodes)
    session.add_all(all_edges)
    session.commit()
    session.close()
