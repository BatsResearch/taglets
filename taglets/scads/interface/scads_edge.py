class ScadsEdge:
    """
    A class representing a relationship between two nodes in the SCADS
    """
    def __init__(self, start_node, end_node, relationship, directed):
        self.start_node = start_node        # The starting ScadsNode in the edge
        self.end_node = end_node            # The ending ScadsNode in the edge
        self.relationship = relationship    # The type of relationship
        self.directed = directed      # Whether or not the relationship is directed

    def get_end_node(self):
        return self.end_node

    def get_relationship(self):
        return self.relationship

    def is_directed(self):
        return self.directed
