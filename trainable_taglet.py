from taglet import Taglet


class TrainableTaglet(Taglet):
    """
    Trainable Taglet class
    """
    def train(self):
        """
        Training the model
        """
        raise NotImplementedError()
