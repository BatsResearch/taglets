from scads import Scads, add_conceptnet, add_datasets


def create_scads():
    """
    Create the SCADS and add datasets to the database.
    :return: None
    """
    add_conceptnet()
    add_datasets()


def scads_example():
    """
    An example using the SCADS interface.
    """
    Scads.open()                                    # Start the session
    node = Scads.get_node('/c/ab/агыруа/n')         # Get a node
    print(node.get_datasets())                      # Get list of dataset names
    print(node.get_images())                        # Get list of image paths
    outgoing_edges = node.get_neighbors()           # Get node neighbors
    print(outgoing_edges[0].relationship)           # Get the type of relationship
    neighboring_node = outgoing_edges[0].end_node   # Get the neighboring node

    # The neighbor is also a ScadsNode
    print(neighboring_node.get_datasets(), neighboring_node.get_images(), neighboring_node.get_neighbors())
    Scads.close()                                   # Close the session


def main():
    create_scads()
    scads_example()


if __name__ == "__main__":
    main()
