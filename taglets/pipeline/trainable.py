import copy
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import torch
import torch.multiprocessing as mp
from accelerate import Accelerator
accelerator = Accelerator()

log = logging.getLogger(__name__)


class Trainable:
    """
    A class with a trainable model.

    Anything that might run on a GPU and/or with multiprocessing should inherit
    from this class.
    """

    def __init__(self, task):
        """
        Create a new Trainable.

        :param task: The current task
        """
        self.name = 'base'
        self.task = task
        self.lr = 0.0005
        self.criterion = torch.nn.CrossEntropyLoss()
        self.seed = 0
        self.num_epochs = 30 if not os.environ.get("CI") else 5
        self.batch_size = task.batch_size if not os.environ.get("CI") else 32
        self.select_on_val = True  # If true, save model on the best validation performance
        self.save_dir = None
        # for extra flexibility
        self.unlabeled_batch_size = self.batch_size

        n_gpu = torch.cuda.device_count()
        self.n_proc = n_gpu if n_gpu > 0 else max(1, mp.cpu_count() - 1)
        self.num_workers = min(max(0, mp.cpu_count() // self.n_proc - 1), 2) if not os.environ.get("CI") else 0

        self.model = task.get_initial_model()

        self._init_random(self.seed)

        # Parameters needed to be updated based on freezing layer
        params_to_update = []
        for param in self.model.parameters():
            if param.requires_grad:
                params_to_update.append(param)
        self._params_to_update = params_to_update
        self.optimizer = torch.optim.Adam(self._params_to_update, lr=self.lr, weight_decay=1e-4)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.1)
        self.valid = True

    def train(self, train_data, val_data, unlabeled_data=None):
        self._do_train(train_data, val_data, unlabeled_data)

    def predict(self, data):
        return self._do_predict(data)

    def evaluate(self, labeled_data):
        """
        Evaluate on labeled data.

        :param labeled_data: A Dataset containing images and ground truth labels
        :return: accuracy
        """
        outputs, labels = self.predict(labeled_data)
        correct = (np.argmax(outputs, 1) == labels).sum()
        return correct / outputs.shape[0]

    def _get_dataloader(self, data, shuffle, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        return accelerator.prepare(torch.utils.data.DataLoader(
            dataset=data, batch_size=batch_size, shuffle=shuffle,
            num_workers=self.num_workers, pin_memory=True
        ))

    def _get_pred_classifier(self):
        return self.model

    @staticmethod
    def save_plot(plt_mode, val_dic, save_dir):
        plt.figure()
        colors = ['r', 'b', 'g']

        counter = 0
        for k, v in val_dic.items():
            val = [np.round(float(i), decimals=3) for i in v]
            plt.plot(val, color=colors[counter], label=k + ' ' + plt_mode)
            counter += 1

        if plt_mode == 'loss':
            plt.legend(loc='upper right')
        elif plt_mode == 'accuracy':
            plt.legend(loc='lower right')
        title = '_vs.'.join(list(val_dic.keys()))
        plt.title(title + ' ' + plt_mode)
        plt.savefig(save_dir + '/' + plt_mode + '_' + title + '.pdf')
        plt.close()

    @staticmethod
    def _init_random(seed):
        """
        Initialize random numbers with a seed.
        :param seed: The seed to initialize with
        :return: None
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def _do_train(self, train_data, val_data, unlabeled_data=None):
        """
        One worker for training.

        This method carries out the actual training iterations. It is designed
        to be called by train().

        :param train_data: A dataset containing training data
        :param val_data: A dataset containing validation data
        :param unlabeled_data: A dataset containing unlabeled data
        :return:
        """
        log.info('Beginning training')
        train_data_loader = self._get_dataloader(data=train_data, shuffle=True)

        if val_data is None:
            val_data_loader = None
        else:
            val_data_loader = self._get_dataloader(data=val_data, shuffle=False)

        if unlabeled_data is None:
            unlabeled_data_loader = None
        else:
            unlabeled_data_loader = self._get_dataloader(data=unlabeled_data,
                                                         shuffle=True,
                                                         batch_size=self.unlabeled_batch_size)
        # Initializes statistics containers (will only be filled by lead process)
        best_model_to_save = None
        best_val_acc = 0
        train_loss_list = []
        train_acc_list = []
        val_loss_list = []
        val_acc_list = []

        accelerator.wait_for_everyone()
        self.model, self.optimizer = accelerator.prepare(self.model, self.optimizer)

        # Iterates over epochs
        for epoch in range(self.num_epochs):
            log.info("Epoch {}: ".format(epoch + 1))

            # Trains on training data
            train_loss, train_acc = self._train_epoch(train_data_loader, unlabeled_data_loader)
            # Evaluates on validation data
            if val_data_loader:
                val_loss, val_acc = self._validate_epoch(val_data_loader)
            else:
                val_loss = 0
                val_acc = 0

            log.info('Train loss: {:.4f}'.format(train_loss))
            log.info('Train acc: {:.4f}%'.format(train_acc * 100))
            train_loss_list.append(train_loss)
            train_acc_list.append(train_acc)
            log.info('Validation loss: {:.4f}'.format(val_loss))
            log.info('Validation acc: {:.4f}%'.format(val_acc * 100))
            val_loss_list.append(val_loss)
            val_acc_list.append(val_acc)
            if val_acc > best_val_acc:
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(self.model)
                
                log.debug("Deep copying new best model." +
                          "(validation of {:.4f}%, over {:.4f}%)".format(
                              val_acc * 100, best_val_acc * 100))
                best_model_to_save = copy.deepcopy(unwrapped_model.state_dict())
                best_val_acc = val_acc
                if self.save_dir:
                    accelerator.save(best_model_to_save, self.save_dir + '/model.pth.tar')

            if self.lr_scheduler:
                self.lr_scheduler.step()

        accelerator.wait_for_everyone()
        self.optimizer = self.optimizer.optimizer
        self.model = accelerator.unwrap_model(self.model)
        self.model.cpu()
        accelerator.free_memory()

        if self.save_dir and accelerator.is_local_main_process:
            val_dic = {'train': train_loss_list, 'validation': val_loss_list}
            self.save_plot('loss', val_dic, self.save_dir)
            val_dic = {'train': train_acc_list, 'validation': val_acc_list}
            self.save_plot('accuracy', val_dic, self.save_dir)
        if self.select_on_val and best_model_to_save is not None:
            self.model.load_state_dict(best_model_to_save)
            accelerator.wait_for_everyone()

    def _train_epoch(self, train_data_loader, unlabeled_data_loader=None):
        raise NotImplementedError

    def _validate_epoch(self, val_data_loader):
        raise NotImplementedError

    @staticmethod
    def _get_train_acc(outputs, labels):
        """
        Gets the training accuracy for a tensor of outputs and training labels.

        The method primarily exists so that EndModel can compute the accuracy of
        soft labels differently.

        :param outputs: outputs of the model being trained
        :param labels: training labels
        :return: the total number of correct predictions
        """
        return torch.sum(torch.max(outputs, 1)[1] == labels)

    def _do_predict(self, data):
        log.info('Beginning prediction')
        pred_classifier = self._get_pred_classifier()
        pred_classifier.eval()
        
        #log.info(f"Data dataLoader {data.filepaths}")
        data_loader = self._get_dataloader(data, False)
        
        accelerator.wait_for_everyone()
        self.model = accelerator.prepare(self.model)
        
        outputs, labels = self._predict_epoch(data_loader, pred_classifier)

        self.model = accelerator.unwrap_model(self.model)
        # Top: Not 100% sure why we need the following line, but if removed, there will be some leftover gpu tensors
        # taking up some memory at the next checkpoint
        accelerator.free_memory()
        
        if len(labels) > 0:
            return outputs, labels
        else:
            return outputs
            
    def _predict_epoch(self, data_loader, pred_classifier):
        raise NotImplementedError

    @staticmethod
    def _get_model_output_shape(in_size, mod):
        """
        Adopt from https://gist.github.com/lebedov/0db63ffcd0947c2ea008c4a50be31032
        Compute output size of Module `mod` given an input with size `in_size`
        :param in_size: input shape (height, width)
        :param mod: PyTorch model
        :return:
        """
        mod = mod.cpu()
        f = mod(torch.rand(2, 3, *in_size))
        return int(np.prod(f.size()[1:]))
    
    
class ImageTrainable(Trainable):
    def _train_epoch(self, train_data_loader, unlabeled_data_loader=None):
        """
        Train for one epoch.
        :param train_data_loader: A dataloader containing training data
        :param use_gpu: Whether or not to use the GPU
        :return: None
        """
        
        self.model.train()
        running_loss = 0
        running_acc = 0
        total_len = 0
        for batch in train_data_loader:
            inputs = batch[0]
            labels = batch[1]

            self.optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                accelerator.backward(loss)
                self.optimizer.step()

            outputs = accelerator.gather(outputs.detach())
            labels = accelerator.gather(labels)

            running_loss += loss.item()
            running_acc += self._get_train_acc(outputs, labels).item()
            total_len += len(labels)

        if not len(train_data_loader):
            return 0, 0

        epoch_loss = running_loss / len(train_data_loader)
        epoch_acc = running_acc / total_len

        return epoch_loss, epoch_acc
    
    def _validate_epoch(self, val_data_loader):
        """
        Validate for one epoch.
        :param val_data_loader: A dataloader containing validation data
        :param use_gpu: Whether or not to use the GPU
        :return: None
        """
        self.model.eval()
        running_loss = 0
        running_acc = 0
        total_len = 0
        for batch in val_data_loader:
            inputs = batch[0]
            labels = batch[1]
            with torch.set_grad_enabled(False):
                outputs = self.model(inputs)
                loss = torch.nn.functional.cross_entropy(outputs, labels)
                _, preds = torch.max(outputs, 1)
            
            preds = accelerator.gather(preds.detach())
            labels = accelerator.gather(labels)

            running_loss += loss.item()
            running_acc += torch.sum(preds == labels).item()
            total_len += len(labels)

        epoch_loss = running_loss / len(val_data_loader)
        epoch_acc = running_acc / total_len

        return epoch_loss, epoch_acc
    
    def _predict_epoch(self, data_loader, pred_classifier):
        outputs = []
        labels = []
        for batch in data_loader:
            if isinstance(batch, list):
                inputs, targets = batch
            else:
                inputs, targets = batch, None
            
            with torch.set_grad_enabled(False):
                output = pred_classifier(inputs)
                outputs.append(torch.nn.functional.softmax(accelerator.gather(output.detach()).cpu(), 1))
                if targets is not None:
                    labels.append(accelerator.gather(targets.detach()).cpu())
        
        outputs = torch.cat(outputs).numpy()
        if len(labels) > 0:
            labels = torch.cat(labels).numpy()
            
        # Accelerate pads the dataset if its length is not divisible by the "actual" batch size
        # so we need to remove the extra elements
        dataset_len = len(data_loader.dataset)
        outputs = outputs[:dataset_len]
        labels = labels[:dataset_len]
        
        return outputs, labels
        
        
class VideoTrainable(Trainable):
    def _train_epoch(self, train_data_loader, unlabeled_data_loader=None):
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
            inputs = inputs["video"]
            #inputs = [i.to(device)[None, ...] for i in inputs]
            #num_videos = inputs.size(0)
            #num_frames = inputs.size(1)
            #inputs = inputs.flatten(start_dim=0, end_dim=1)
            
            self.optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = self.model(inputs)
                # aggregated_outputs = torch.mean(outputs.view(num_videos, num_frames, -1), dim=1)
                loss = self.criterion(outputs, labels)
                accelerator.backward(loss)
                self.optimizer.step()

            aggregated_outputs = accelerator.gather(outputs.detach())
            labels = accelerator.gather(labels)
            
            running_loss += loss.item()
            running_acc += self._get_train_acc(aggregated_outputs, labels).item()
        
        if not len(train_data_loader.dataset):
            return 0, 0
        
        epoch_loss = running_loss / len(train_data_loader)
        epoch_acc = running_acc / len(train_data_loader.dataset)
        
        return epoch_loss, epoch_acc
    
    def _validate_epoch(self, val_data_loader):
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
            inputs = inputs["video"]
            #num_videos = inputs.size(0)
            #num_frames = inputs.size(1)
            #inputs = inputs.flatten(start_dim=0, end_dim=1)
            
            with torch.set_grad_enabled(False):
                outputs = self.model(inputs)
                #log.info(outputs)
                #aggregated_outputs = torch.mean(outputs.view(num_videos, num_frames, -1), dim=1)
                loss = self.criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

            preds  = accelerator.gather(preds.detach())
            labels = accelerator.gather(labels)

            running_loss += loss.item()
            running_acc += torch.sum(preds == labels).item()
        
        epoch_loss = running_loss / len(val_data_loader.dataset)
        epoch_acc = running_acc / len(val_data_loader.dataset)
        
        return epoch_loss, epoch_acc
    
    def _predict_epoch(self, data_loader, pred_classifier):
        outputs = []
        labels = []
        for batch in data_loader:
            #log.info(batch)
            if isinstance(batch, list):
                inputs, targets = batch
            else:
                inputs, targets = batch, None
            
            #num_videos = inputs.size(0)
            #num_frames = inputs.size(1)
            #inputs = inputs.flatten(start_dim=0, end_dim=1)
            inputs = inputs["video"] 
            
            with torch.set_grad_enabled(False):
                output = pred_classifier(inputs)
                #aggregated_output = torch.mean(output.view(num_videos, num_frames, -1), dim=1)
                #outputs.append(torch.nn.functional.softmax(accelerator.gather(aggregated_output.detach()).cpu(), 1))
                outputs.append(accelerator.gather(output.detach()).cpu())
                if targets is not None:
                    labels.append(accelerator.gather(targets.detach()).cpu())
        
        outputs = torch.cat(outputs).numpy()
        if len(labels) > 0:
            labels = torch.cat(labels).numpy()

        # Accelerate pads the dataset if its length is not divisible by the "actual" batch size
        # so we need to remove the extra elements
        dataset_len = len(data_loader.dataset)
        outputs = outputs[:dataset_len]
        labels = labels[:dataset_len]
        
        return outputs, labels
