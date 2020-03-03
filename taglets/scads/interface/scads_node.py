from .scads_edge import ScadsEdge
from ..create.scads_classes import Node


class ScadsNode:
    """
    A class representing a node in the SCADS.
    """
    def __init__(self, node, session):
        self.node = node
        self.session = session

    def get_datasets(self):
        """
        Get list of datasets.
        :return: List of our datasets containing the node
        """
        datasets = set()
        images = self.node.images
        for image in images:
            datasets.add(image.dataset.name)
        return list(datasets)

    def get_images(self):
        """
        Get all paths to images for this concept.
        :return: List of paths to images for this concept
        """
        locations = []
        for image in self.node.images:
            locations.append(image.path)
        return locations

    def get_neighbors(self):
        """
        Get the neighbors of this concept with the type of relationship.
        :return: List of ScadsEdges
        """
        edges = []
        for edge in self.node.outgoing_edges:
            sql_node = self.session.query(Node).filter(Node.id == edge.end_node).first()
            end_node = ScadsNode(sql_node, self.session)
            edges.append(ScadsEdge(self,
                                   end_node,
                                   edge.relation.type,
                                   edge.relation.is_directed))
        return edges
