class ScadsEdge:
    """
    A class to represent a relationship between two nodes in the SCADS
    """
    def __init__(self, node, relationship):
        self.node = node                    # The other node in the edge
        self.relationship = relationship    # The type of relationship
