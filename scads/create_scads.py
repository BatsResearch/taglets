from scads.scads_classes import *
import os


def get_relations():
    """ load the csv file containing all relations in conceptnet, and return a list of Relation class."""

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
    """ load the csv file containing all nodes in conceptnet, and return a list of 'Node' class"""

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
    """ load the csv file containing all edges in conceptnet, and return a list of 'Edge' class."""

    cwd = os.getcwd()
    all_edges = []
    with open(os.path.join(cwd, 'sql_data', 'edges.csv'), encoding="utf8") as edge_file:
        for line in edge_file:
            if line.strip():
                key, URI, relation_key, start_node_key, end_node_key,weight = line.strip().split()[:6]
                info = " " .join(line.strip().split()[6:])
                edge = Edge(key=key,URI=str(URI),relation_key=relation_key, start_node_key=start_node_key,
                                end_node_key=end_node_key, weight=weight, info=info)
                all_edges.append(edge)
    return all_edges


def add_conceptnet():
    """Insert conceptnet information such as nodes, edges, and relations to correspoinding tables. """

    ########## Generate database schema #########
    Base.metadata.create_all(engine)

    session = Session()

    ########## Read nodes, edges, and relations from files and insert into corresponding tables #########
    all_relations = get_relations()
    all_nodes = get_nodes()
    all_edges = get_edges()

    # Top: this assumes the relations in all_relations are already SORTED based on their keys and the keys start at 0
    for edge in all_edges:
        edge.relation = all_relations[int(edge.relation_key)]
        edge.end_node = all_nodes[int(edge.end_node_key)]

    session.add_all(all_relations)
    session.add_all(all_nodes)
    session.add_all(all_edges)
    session.commit()
    session.close()
