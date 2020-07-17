import logging
from .trainable import Trainable
import torch

log = logging.getLogger(__name__)


class Taglet(Trainable):
    """
    A trainable model that produces votes for unlabeled images
    """
    def execute(self, unlabeled_data, use_gpu):
        """
        Execute the Taglet on unlabeled images.

        :param unlabeled_data_loader: A dataloader containing unlabeled data
        :param use_gpu: Whether or not the use the GPU
        :return: A list of predicted labels
        """
        unlabeled_data_loader = torch.utils.data.DataLoader(
            dataset=unlabeled_data, batch_size=self.batch_size, shuffle=False,
            num_workers=1, pin_memory=True
        )

        self.model.eval()
        if use_gpu:
            self.model = self.model.cuda()
        else:
            self.model = self.model.cpu()

        predicted_labels = []
        for inputs in unlabeled_data_loader:
            if use_gpu:
                inputs = inputs.cuda()
            with torch.set_grad_enabled(False):
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                predicted_labels = predicted_labels + preds.detach().cpu().tolist()
        return predicted_labels
