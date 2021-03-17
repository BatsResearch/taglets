from .taglet import Trainable

import logging
import os
import torch

log = logging.getLogger(__name__)


class EndModel(Trainable):
    def __init__(self, task):
        super().__init__(task)
        self.name = 'end model'
        m = torch.nn.Sequential(*list(self.model.children())[:-1])
        output_shape = self._get_model_output_shape(self.task.input_shape, m)
        self.model.fc = torch.nn.Sequential(torch.nn.Dropout(0.3),
                                            torch.nn.Linear(output_shape, len(self.task.classes)))
        if os.getenv("LWLL_TA1_PROB_TASK") is not None:
            self.save_dir = os.path.join('/home/tagletuser/trained_models', self.name)
        else:
            self.save_dir = os.path.join('trained_models', self.name)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.criterion = self.soft_cross_entropy

        params_to_update = []
        for param in self.model.parameters():
            if param.requires_grad:
                params_to_update.append(param)
        self._params_to_update = params_to_update
        self.optimizer = torch.optim.Adam(self._params_to_update, lr=self.lr, weight_decay=1e-4)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.1)

    @staticmethod
    def soft_cross_entropy(outputs, target):
        outputs = outputs.double()
        target = target.double()
        logs = torch.nn.LogSoftmax(dim=1)
        return torch.mean(torch.sum(-target * logs(outputs), 1))

    @staticmethod
    def _get_train_acc(outputs, labels):
        return torch.sum(torch.max(outputs, 1)[1] == torch.max(labels, 1)[1])
    
    
class VideoEndModel(EndModel):
    def _train_epoch(self, rank, train_data_loader, unlabeled_data_loader=None):
        """
        Train for one epoch.
        :param train_data_loader: A dataloader containing training videos
        :param use_gpu: Whether or not to use the GPU
        :return: None
        """
        self.model.train()
        running_loss = 0
        running_acc = 0
        for batch in train_data_loader:
            inputs = batch[0]
            labels = batch[1]
            if self.use_gpu:
                inputs = inputs.cuda(rank)
                labels = labels.cuda(rank)
            num_videos = inputs.size(0)
            num_frames = inputs.size(1)
            inputs = inputs.flatten(start_dim=0, end_dim=1)

            self.optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = self.model(inputs)
                aggregated_outputs = torch.mean(outputs.view(num_videos, num_frames, -1), dim=1)
                loss = self.criterion(aggregated_outputs, labels)
                loss.backward()
                self.optimizer.step()

            running_loss += loss.item()
            running_acc += self._get_train_acc(aggregated_outputs, labels)

        if not len(train_data_loader.dataset):
            return 0, 0

        epoch_loss = running_loss / len(train_data_loader.dataset)
        epoch_acc = running_acc.item() / len(train_data_loader.dataset)

        return epoch_loss, epoch_acc
    
    def _validate_epoch(self, rank, val_data_loader,):
        """
        Train for one epoch.
        :param train_data_loader: A dataloader containing training videos
        :param use_gpu: Whether or not to use the GPU
        :return: None
        """
        self.model.eval()
        running_loss = 0
        running_acc = 0
        for batch in val_data_loader:
            inputs = batch[0]
            labels = batch[1]
            if self.use_gpu:
                inputs = inputs.cuda(rank)
                labels = labels.cuda(rank)
            num_videos = inputs.size(0)
            num_frames = inputs.size(1)
            inputs = inputs.flatten(start_dim=0, end_dim=1)

            with torch.set_grad_enabled(False):
                outputs = self.model(inputs)
                aggregated_outputs = torch.mean(outputs.view(num_videos, num_frames, -1), dim=1)
                loss = self.criterion(aggregated_outputs, labels)
                _, preds = torch.max(aggregated_outputs, 1)

            running_loss += loss.item()
            running_acc += self._get_train_acc(aggregated_outputs, labels)

        epoch_loss = running_loss / len(val_data_loader.dataset)
        epoch_acc = running_acc.item() / len(val_data_loader.dataset)

        return epoch_loss, epoch_acc

    def _do_predict(self, rank, q, data):
        if rank == 0:
            log.info('Beginning prediction')

        pred_classifier = self._get_pred_classifier()
        pred_classifier.eval()

        # Configures model for device
        if self.use_gpu:
            pred_classifier = pred_classifier.cuda(rank)
        else:
            pred_classifier = pred_classifier.cpu()

        # Creates distributed data loader from dataset
        sampler = torch.utils.data.distributed.DistributedSampler(
            data, num_replicas=self.n_proc, rank=rank, shuffle=False
        )
        data_loader = torch.utils.data.DataLoader(
            dataset=data, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True, sampler=sampler
        )

        outputs = []
        labels = []
        for batch in data_loader:
            if isinstance(batch, list):
                if len(batch) == 2:
                    inputs, targets = batch
                else:
                    inputs, targets = batch[0], None
            else:
                inputs, targets = batch, None

            if self.use_gpu:
                inputs = inputs.cuda(rank)
            num_videos = inputs.size(0)
            num_frames = inputs.size(1)
            inputs = inputs.flatten(start_dim=0, end_dim=1)

            with torch.set_grad_enabled(False):
                output = pred_classifier(inputs)
                aggregated_output = torch.mean(output.view(num_videos, num_frames, -1), dim=1)
                outputs.append(torch.nn.functional.softmax(aggregated_output, 1))
                if targets is not None:
                    labels.append(targets)

        outputs = torch.cat(outputs).cpu().detach().numpy()
        if len(labels) > 0:
            labels = torch.cat(labels).cpu().detach().numpy()

        if rank == 0:
            log.info('Finished prediction')

        if len(labels) > 0:
            q.put((rank, outputs, labels))
        else:
            q.put((rank, outputs))
