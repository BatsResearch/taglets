import copy
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import random
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

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
        self.batch_size = 256 if not os.environ.get("CI") else 32
        self.select_on_val = True  # If true, save model on the best validation performance
        self.save_dir = None
        # for extra flexibility
        self.unlabeled_batch_size = self.batch_size

        # Configures GPU and multiprocessing
        n_gpu = torch.cuda.device_count()
        if n_gpu > 0:
            self.use_gpu = True
            self.n_proc = n_gpu
        else:
            self.use_gpu = False
            self.n_proc = max(1, mp.cpu_count() - 1)
        self.num_workers = min(max(0, mp.cpu_count() // self.n_proc - 1), 2)

        # Gradients are summed over workers, so need to scale the step size
        self.lr = self.lr / self.n_proc

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
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '8888'
        # Launches workers and collects results from queue
        processes = []
        ctx = mp.get_context('spawn')
        q = ctx.Queue()
        for i in range(self.n_proc):
            args = (i, q, train_data, val_data, unlabeled_data)
            p = ctx.Process(target=self._do_train, args=args)
            p.start()
            processes.append(p)

        state_dict = q.get()

        for p in processes:
            p.join()

        state_dict = pickle.loads(state_dict)
        self.model.load_state_dict(state_dict)

    def predict(self, data):
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '8888'

        # Launches workers and collects results from queue
        processes = []
        results = []
        ctx = mp.get_context('spawn')
        q = ctx.Queue()
        for i in range(self.n_proc):
            args = (i, q, data)
            p = ctx.Process(target=self._do_predict, args=args)
            p.start()
            processes.append(p)
        for _ in processes:
            results.append(q.get())
        for p in processes:
            p.join()

        # Collates results so output order matches input order
        # Results are either (rank, outputs) or (rank, outputs, label)
        #
        # Remember to only return the first len(data) results, since the
        # distributed data sampler adds extra copies to make each worker the same
        outputs = np.ndarray((len(data), results[0][1].shape[1]), dtype=np.float32)
        if len(results[0]) > 2:
            labels = np.ndarray((len(data),), dtype=np.int32)
        else:
            labels = None

        # Sorts results in rank order
        results.sort(key=lambda x: x[0])

        # Collects the results
        for i in range(len(data)):
            outputs[i] = results[i % len(results)][1][i // len(results), :]
            if labels is not None:
                labels[i] = results[i % len(results)][2][i // len(results)]

        if labels is not None:
            return outputs, labels
        else:
            return outputs

    def evaluate(self, labeled_data):
        """
        Evaluate on labeled data.

        :param labeled_data: A Dataset containing images and ground truth labels
        :return: accuracy
        """
        outputs, labels = self.predict(labeled_data)
        correct = (np.argmax(outputs, 1) == labels).sum()
        return correct / outputs.shape[0]

    def _get_train_sampler(self, data, n_proc, rank):
        return torch.utils.data.distributed.DistributedSampler(data,
                                                               num_replicas=n_proc,
                                                               rank=rank)

    def _get_val_sampler(self, data, n_proc, rank):
        return self._get_train_sampler(data, n_proc, rank)

    def _get_dataloader(self, data, sampler, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        return torch.utils.data.DataLoader(
            dataset=data, batch_size=batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True, sampler=sampler
        )

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

    def _do_train(self, rank, q, train_data, val_data, unlabeled_data=None):
        """
        One worker for training.

        This method carries out the actual training iterations. It is designed
        to be called by train().

        :param train_data: A dataset containing training data
        :param val_data: A dataset containing validation data
        :param unlabeled_data: A dataset containing unlabeled data
        :return:
        """
        if rank == 0:
            log.info('Beginning training')

        # Initializes distributed backend
        backend = 'nccl' if self.use_gpu else 'gloo'
        dist.init_process_group(
            backend=backend, init_method='env://', world_size=self.n_proc, rank=rank
        )

        # Configures model to be distributed
        if self.use_gpu:
            self.model = self.model.cuda(rank)
            self.model = nn.parallel.DistributedDataParallel(
                self.model, device_ids=[rank]
            )
        else:
            self.model = self.model.cpu()
            self.model = nn.parallel.DistributedDataParallel(
                self.model, device_ids=None
            )


        # Creates distributed data loaders from datasets
        train_sampler = self._get_train_sampler(train_data, n_proc=self.n_proc, rank=rank)
        train_data_loader = self._get_dataloader(data=train_data, sampler=train_sampler)

        if val_data is None:
            val_data_loader = None
        else:
            val_sampler = self._get_val_sampler(val_data, n_proc=self.n_proc, rank=rank)
            val_data_loader = self._get_dataloader(data=val_data, sampler=val_sampler)

        if unlabeled_data is None:
            unlabeled_data_loader = None
        else:
            unlabeled_sampler = self._get_train_sampler(data=unlabeled_data,
                                                            n_proc=self.n_proc,
                                                            rank=rank)
            unlabeled_data_loader = self._get_dataloader(data=unlabeled_data,
                                                         sampler=unlabeled_sampler,
                                                         batch_size=self.unlabeled_batch_size)
        # Initializes statistics containers (will only be filled by lead process)
        best_model_to_save = None
        best_val_acc = 0
        train_loss_list = []
        train_acc_list = []
        val_loss_list = []
        val_acc_list = []

        # Iterates over epochs
        for epoch in range(self.num_epochs):
            if rank == 0:
                log.info("Epoch {}: ".format(epoch + 1))

            # this is necessary for shuffle to work
            train_sampler.set_epoch(epoch)

            # Trains on training data
            train_loss, train_acc = self._train_epoch(rank, train_data_loader,
                                                            unlabeled_data_loader)
            # Evaluates on validation data
            if val_data_loader:
                val_loss, val_acc = self._validate_epoch(rank, val_data_loader)
            else:
                val_loss = 0
                val_acc = 0

            # Gathers results statistics to lead process
            summaries = [train_loss, train_acc, val_loss, val_acc]
            summaries = torch.tensor(summaries, requires_grad=False)
            if self.use_gpu:
                summaries = summaries.cuda(rank)
            else:
                summaries = summaries.cpu()
            dist.reduce(summaries, 0, op=dist.ReduceOp.SUM)
            train_loss, train_acc, val_loss, val_acc = summaries

            # Processes results if lead process
            if rank == 0:
                log.info('Train loss: {:.4f}'.format(train_loss))
                log.info('Train acc: {:.4f}%'.format(train_acc * 100))
                train_loss_list.append(train_loss)
                train_acc_list.append(train_acc)
                log.info('Validation loss: {:.4f}'.format(val_loss))
                log.info('Validation acc: {:.4f}%'.format(val_acc * 100))
                val_loss_list.append(val_loss)
                val_acc_list.append(val_acc)
                if val_acc > best_val_acc:
                    log.debug("Deep copying new best model." +
                              "(validation of {:.4f}%, over {:.4f}%)".format(
                                  val_acc * 100, best_val_acc * 100))
                    best_model_to_save = copy.deepcopy(self.model.module.state_dict())
                    best_val_acc = val_acc
                    if self.save_dir:
                        torch.save(best_model_to_save, self.save_dir + '/model.pth.tar')

            if self.lr_scheduler:
                self.lr_scheduler.step()

        # Lead process saves plots and returns best model
        if rank == 0:
            if self.save_dir:
                val_dic = {'train': train_loss_list, 'validation': val_loss_list}
                self.save_plot('loss', val_dic, self.save_dir)
                val_dic = {'train': train_acc_list, 'validation': val_acc_list}
                self.save_plot('accuracy', val_dic, self.save_dir)
            if self.select_on_val and best_model_to_save is not None:
                self.model.module.load_state_dict(best_model_to_save)

            self.model.cpu()
            state_dict = self.model.module.state_dict()
            state_dict = pickle.dumps(state_dict)
            q.put(state_dict)

        # Use a barrier to keep all workers alive until they all finish,
        # due to shared CUDA tensors. See
        # https://pytorch.org/docs/stable/multiprocessing.html#multiprocessing-cuda-sharing-details
        dist.barrier()

    def _train_epoch(self, rank, train_data_loader, unlabeled_data_loader=None):
        """
        Train for one epoch.
        :param train_data_loader: A dataloader containing training data
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

            self.optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

            running_loss += loss.item()
            running_acc += self._get_train_acc(outputs, labels)

        if not len(train_data_loader.dataset):
            return 0, 0

        epoch_loss = running_loss / len(train_data_loader.dataset)
        epoch_acc = running_acc.item() / len(train_data_loader.dataset)

        return epoch_loss, epoch_acc

    def _validate_epoch(self, rank, val_data_loader):
        """
        Validate for one epoch.
        :param val_data_loader: A dataloader containing validation data
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
            with torch.set_grad_enabled(False):
                outputs = self.model(inputs)
                loss = torch.nn.functional.cross_entropy(outputs, labels)
                _, preds = torch.max(outputs, 1)

            running_loss += loss.item()
            running_acc += torch.sum(preds == labels)

        epoch_loss = running_loss / len(val_data_loader.dataset)
        epoch_acc = running_acc.item() / len(val_data_loader.dataset)

        return epoch_loss, epoch_acc

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
                inputs, targets = batch
            else:
                inputs, targets = batch, None

            if self.use_gpu:
                inputs = inputs.cuda(rank)

            with torch.set_grad_enabled(False):
                output = pred_classifier(inputs)
                outputs.append(torch.nn.functional.softmax(output, 1))
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
        