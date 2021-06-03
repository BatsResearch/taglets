from .scads_edge import ScadsEdge
from ..create.scads_classes import Node


class ScadsNode:
    """
    A class representing a node in the SCADS.
    """
    def __init__(self, node, session):
        self.node = node
        self.session = session

    def get_conceptnet_id(self):
        return self.node.conceptnet_id

    def get_datasets(self, images=True):
        """
        Get list of datasets.
        :return: List of our datasets containing the node
        """
        datasets = set()
        data = self.node.images if images else self.node.clips
        for datum in data:
            datasets.add(datum.dataset.name)
        return list(datasets)

    def get_images(self):
        """
        Get all paths to images for this concept.
        :return: List of paths to images for this concept
        """
        q = "SELECT DISTINCT images.path from images " \
            "JOIN nodes ON images.node_id = nodes.id " \
            "JOIN datasets ON images.dataset_id = datasets.id " \
            "WHERE nodes.id = " + str(self.node.id) + ";"

        results = self.session.connection().execute(q)
        return [x[0] for x in results]

    def get_images_whitelist(self, whitelist):
        """
        Get all paths to the white list images for this concept .
        :return: List of paths to images for this concept
        """
        if whitelist is not None and len(whitelist) > 0:
            q = "SELECT images.path from images " \
                "JOIN nodes ON images.node_id = nodes.id " \
                "JOIN datasets ON images.dataset_id = datasets.id " \
                "WHERE nodes.id = " + str(self.node.id) \
                + " AND (datasets.path = '" + whitelist[0] + "'"
            for path in whitelist[1:]:
                q += " OR datasets.path = '" + path + "'"
            q += ");"
            results = self.session.connection().execute(q)
            return [x[0] for x in results]
        else:
            return self.get_images()

    def get_clips(self):
        q = "SELECT DISTINCT base_path, start_frame, end_frame, clip_id, video_id from clips " \
            "JOIN nodes ON clips.node_id=nodes.id " \
            "WHERE nodes.id = " + str(self.node.id) + ";"
        results = self.session.connection().execute(q)
        return [(x[0], x[1], x[2]) for x in results]

    def get_clips_whitelist(self, whitelist):
        if whitelist is not None and len(whitelist) > 0:
            q = "SELECT DISTINCT clips.base_path, start_frame, end_frame, clip_id, video_id from clips " \
                "JOIN nodes ON clips.node_id = nodes.id " \
                "JOIN datasets ON images.dataset_id = datasets.id " \
                "WHERE nodes.id = " + str(self.node.id) \
                + " AND (datasets.path = '" + whitelist[0] + "'"
            for path in whitelist[1:]:
                q += " OR datasets.path = '" + path + "'"
            q += ");"
            results = self.session.connection().execute(q)
            return [(x[0], x[1], x[2]) for x in results]
        else:
            return self.get_clips()

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
                                   edge.weight,
                                   edge.relation.is_directed))
        return edges
 