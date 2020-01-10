from scads import Scads

from modules.module import BaseModule
from taglet_executer import TagletExecuter
from task import MNIST


def scads_example():
    Scads.open()                                    # Start the session
    node = Scads.get_node('/c/ab/агыруа/n')         # Get a node
    print(node.get_datasets())                      # Get list of dataset names
    print(node.get_images())                        # Get list of image paths
    outgoing_edges = node.get_neighbors()           # Get node neighbors
    print(outgoing_edges[0].relationship)           # Get the type of relationship
    neighboring_node = outgoing_edges[0].end_node   # Get the neighboring node

    # The neighbor is also a ScadsNode
    print(neighboring_node.get_datasets(), neighboring_node.get_images(), neighboring_node.get_neighbors)
    Scads.close()                                   # Close the session


def workflow():
    MNIST_task = MNIST()
    MNIST_module = BaseModule(task=MNIST_task)
    MNIST_module.train_taglets(MNIST_task.labeled_images)
    taglets = MNIST_module.get_taglets()
    taglet_executer = TagletExecuter(taglets)
    label_matrix = taglet_executer.execute(MNIST_task.unlabeled_images)
    # soft_labels = LabelModel.annotate(label_matrix)
    # end_model = end_model(soft_labels, self.unlabeled_images)
    # [test_predictions] = end_model.prediction(end_model, self.test_images)
