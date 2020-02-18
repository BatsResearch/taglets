class ScadsEdge:
    """
    A class representing a relationship between two nodes in the SCADS
    """
    def __init__(self, start_node, end_node, relationship, is_directed):
        self.start_node = start_node        # The starting ScadsNode in the edge
        self.end_node = end_node            # The ending ScadsNode in the edge
        self.relationship = relationship    # The type of relationship
        self.is_directed = is_directed      # Whether or not the relationship is directed
