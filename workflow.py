from scads import Scads, add_conceptnet, add_datasets
from modules.module import BaseModule
from taglet_executer import TagletExecuter
from task import Task
from JPL_interface import JPL


def create_scads():
    add_conceptnet()
    add_datasets()


def scads_example():
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


def workflow():
    JPL_API = JPL()
    tasks = JPL_API.get_available_tasks()
    task_name = tasks[1] #problem_test_image_classification
    task_metadata = JPL_API.get_task_metadata(task_name)
    current_task = Task(task_metadata)


    JPL_API.create_session(current_task)
    session_status = JPL_API.get_session_status()
    current_dataset = session_status['current_dataset']
    current_task.classes = current_dataset.classes
    current_task.unlabeled_images = current_dataset.data_url
    current_task.labeled_images = JPL_API.get_seed_labels()


    MNIST_module = BaseModule(task=current_task)
    MNIST_module.train_taglets(current_task.labeled_images)
    taglets = MNIST_module.get_taglets()
    taglet_executer = TagletExecuter(taglets)
    label_matrix = taglet_executer.execute(current_task.unlabeled_images)
    # soft_labels = LabelModel.annotate(label_matrix)
    # end_model = end_model(soft_labels, self.unlabeled_images)
    # [test_predictions] = end_model.prediction(end_model, self.test_images)


def main():
    create_scads()
    scads_example()


if __name__ == "__main__":
    main()
