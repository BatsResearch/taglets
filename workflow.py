import numpy as np
from pathlib import Path
from modules.module import BaseModule
from taglet_executer import TagletExecuter
from task import Task, MNIST





def workflow():

    MNIST_task = MNIST()
    MNIST_module = BaseModule(task= MNIST_task)
    taglets = MNIST_module.get_taglets()
    taglet_executer = TagletExecuter(taglets)
    taglet_executer.train(MNIST_task.labeled_images)
    label_matrix = taglet_executer.execute(MNIST_task.unlabeled_images)
    # soft_labels = LabelModel.annotate(label_matrix)
    # end_model = end_model(soft_labels, self.unlabeled_images)
    # [test_predictions] = end_model.prediction(end_model, self.test_images)