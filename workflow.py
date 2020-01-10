from modules.module import BaseModule
from taglet_executer import TagletExecuter
from task import MNIST


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
