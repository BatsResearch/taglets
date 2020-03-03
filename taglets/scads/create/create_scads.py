import os
from .scads_classes import Node, Edge, Relation, Base, engine, Session


def get_relations(directory):
    """
    Get all Relations in conceptnet.
    :return: A list of Relations
    """
    cwd = os.getcwd()
    all_relations = []
    with open(os.path.join(cwd, directory, 'relations.csv'), encoding="utf8") as relation_file:
        for line in relation_file:
            if line.strip():
                relation_id, relation_type, directed = line.strip().split()
                is_directed = directed == "t"
                relation = Relation(id=int(relation_id), type=relation_type, is_directed=is_directed)
                all_relations.append(relation)
    return all_relations


def get_nodes(directory):
    """
    Get all Nodes in conceptnet.
    :return: A list of Nodes
    """
    cwd = os.getcwd()
    all_nodes = []
    with open(os.path.join(cwd, directory, 'nodes.csv'), encoding="utf8") as node_file:
        for line in node_file:
            if line.strip():
                node_key, conceptnet_id = line.strip().split()
                if conceptnet_id.startswith("/c/en/"):
                    node = Node(id=node_key, conceptnet_id=conceptnet_id)
                    all_nodes.append(node)
    return all_nodes


def get_edges(directory):
    """
    Get all edges in conceptnet.
    :return: A list of Edges
    """
    cwd = os.getcwd()
    all_edges = []
    with open(os.path.join(cwd, directory, 'edges.csv'), encoding="utf8") as edge_file:
        for line in edge_file:
            if line.strip():
                try:
                    edge_id, _, relation, start_node, end_node = line.strip().split()[:5]
                    edge = Edge(id=int(edge_id),
                                relation_type=int(relation),
                                start_node=int(start_node),
                                end_node=int(end_node))
                    all_edges.append(edge)
                except ValueError:
                    pass
    return all_edges


def add_conceptnet(directory):
    """
    Add Nodes, Edges, and Relations from conceptnet into the database.
    :return: None
    """

    # Read nodes, edges, and relations from files
    all_relations = get_relations(directory)
    all_nodes = get_nodes(directory)
    all_edges = get_edges(directory)

    # Generate database schema
    Base.metadata.create_all(engine)
    session = Session()

    session.add_all(all_relations)
    session.commit()
    session.add_all(all_nodes)
    session.commit()

    included_edges = []
    for edge in all_edges:
        relation = session.query(Relation).filter(Relation.id == edge.relation_type).first()
        start_node = session.query(Node).filter(Node.id == edge.start_node).first()
        end_node = session.query(Node).filter(Node.id == edge.end_node).first()
        if relation and start_node and end_node:
            edge.relation = relation
            start_node.outgoing_edges.append(edge)
            included_edges.append(edge)

    session.add_all(included_edges)

    session.commit()
    session.close()


def main():
    add_conceptnet('../../test_data')


if __name__ == "__main__":
    main()
