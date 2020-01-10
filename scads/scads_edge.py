class ScadsEdge:
    """
    A class to represent a relationship between two nodes in the SCADS
    """
    def __init__(self, start_node, end_node, relationship):
        self.start_node = start_node        # The starting ScadsNode in the edge
        self.end_node = end_node            # The ending ScadsNode in the edge
        self.relationship = relationship    # The type of relationship
