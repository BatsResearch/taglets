# ClassConditionalLabelModule
import numpy as np
from scipy import sparse
import torch
import torch.nn as nn


# A Fork from Label Modals
class ClassConditionalLM(nn.Module):
    def __init__(self,
                 num_classes,
                 num_lfs,
                 acc_prior=0.025,
                 init_acc=0.9,
                 balance_prior=0.025,
                 init_prop=0.5,
                 opt_cb=True):
        super().__init__()

        self.num_classes = num_classes
        self.num_lfs = num_lfs

        # Log Scale Conversion
        # from logistic induction
        self.init_acc = -1 * np.log(1.0 / init_acc - 1) / 2

        # Add accuracies / propensities to Computation Graph

        # Dim: num_lfs * num_classes
        # acc[i, j] ith LF's accuracy over class j
        self.accuracy = nn.Parameter(
            torch.tensor([[self.init_acc] * num_classes for _ in range(num_lfs)]),
            requires_grad=True
        )

        # Dim: num_lfs
        # pro[i]: ith LF's propensity to label (not to specific class!)
        self.propensity = nn.Parameter(
            torch.zeros([num_lfs]),
            requires_grad=True
        )

        self.class_balance = nn.Parameter(
            torch.log(torch.ones([num_classes]) / num_classes),
            requires_grad=opt_cb
        )

        self.balance_prior = balance_prior
        self.acc_prior = acc_prior
        self.trained = False

    def class_conditioned_ll(self, votes):

        # Operates in coordinate triplets format
        if type(votes) != sparse.coo_matrix:
            votes = sparse.coo_matrix(votes)

        num_inst = votes.shape[0]

        # Log-scale, Normalized across row
        class_cond_likelihood = torch.zeros(num_inst, self.num_classes)

        # Normalization factor for propensity
        # Recall: ith lf on class j has
        #
        # a_{i,j} b_{i} if lf votes for that class
        # (1-a_{i,j}) b_{i} if not
        # 1-b_{i} if no votes
        #

        z_prop = self.propensity.unsqueeze(1)
        z_prop = torch.cat((z_prop, torch.zeros((self.num_lfs, 1))), dim=1)
        z_prop = torch.logsumexp(z_prop, dim=1)

        class_cond_likelihood -= torch.sum(z_prop)

        z_acc = self.accuracy.unsqueeze(2)
        z_acc = torch.cat((z_acc, -1 * self.accuracy.unsqueeze(2)), dim=2)
        z_acc = torch.logsumexp(z_acc, dim=2)

        # k instance, j class, i lf
        for k, i, vote in zip(votes.row, votes.col, votes.data):
            for j in range(1, self.num_classes+1):
                if vote == j:
                    # a_{i,j} * b_{i}
                    logprob = self.propensity[i] + self.accuracy[i, j-1] - z_acc[i, j-1]
                elif vote != 0:
                    # Not abstain
                    # (1-a_{i,j}) * b_{i}
                    logprob = self.propensity[i] \
                              - self.accuracy[i, j-1] \
                              - z_acc[i, j-1]
                    logprob -= torch.log(torch.tensor(self.num_classes - 1.0))
                else:
                    # abstain:
                    # logprob = torch.log(1.0 - self.propensity[j])
                    logprob = 0
                class_cond_likelihood[k, j-1] += logprob

        return class_cond_likelihood

    def regularization(self):
        '''
        acc_prior * |accuracy - init_acc|^2 + entropy_prior

        :return:
        '''
        neg_entropy = 0.0
        norm_class_balance = self.class_balance - torch.logsumexp(self.class_balance, dim=0)
        exp_class_balance = torch.exp(norm_class_balance)
        for k in range(self.num_classes):
            neg_entropy += norm_class_balance[k] * exp_class_balance[k]
        entropy_prior = self.balance_prior * neg_entropy

        return self.acc_prior * torch.norm(self.accuracy - self.init_acc) + entropy_prior

    def forward(self, votes):
        '''


        :param votes: num_instances * num_lfs of votes from [0, 1, ..., k], 0->abstantion
        :return: log likelihood
        '''

        class_prior = self.class_balance - torch.logsumexp(self.class_balance, dim=0)
        conditional_ll = self.class_conditioned_ll(votes)
        return torch.logsumexp(class_prior+conditional_ll, dim=1)

    def optimize(self, votes, cfg=None):
        '''


        :param votes: (num_inst, num_lfs)
        :param cfg: dictionary:
            'lr': (float)
            'epoch': (int)
            'seed': (int)
            'batch_size': (int)
            'momentum': (float)
            'step_schedule': (int)
            'step_multiplier': (float)
        '''

        if cfg is None:
            cfg = {'lr': 0.01,
                   'epoch': 10,
                   'seed': 0,
                   'batch_size': 64,
                   'momentum': 0.9,
                   'step_schedule': 10,
                   'step_multiplier': 1.0}

        batch_num = cfg['batch_size']
        votes = sparse.csr_matrix(votes, dtype=np.int)
        batches = self.batchize(votes, batch_num)

        lr = cfg['lr']
        momentum = cfg['momentum']
        step_size_mult = cfg['step_multiplier']
        step_schedule = cfg['step_schedule']
        epochs = cfg['epoch']

        self.init_random(cfg['seed'])

        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=0)

        if step_schedule is not None and step_size_mult is not None:
            LR_milestones = list(
                filter(
                    lambda a: a > 0,
                    [i if (i % step_schedule == 0) else -1 for i in range(epochs)]
                ))
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, LR_milestones, gamma=0.1)
        else:
            scheduler = None

        # for epoch in range(epochs):
        for epoch in range(2):
            # print('Epoch {}/{}'.format(epoch + 1, epochs))
            if scheduler is not None:
                scheduler.step()

            # Sets model to training mode
            self.train()
            running_loss = 0.0

            # Iterates over training data
            for i_batch, inputs in enumerate(batches):
                optimizer.zero_grad()
                log_likelihood = self(*inputs)
                loss = -1 * torch.mean(log_likelihood)
                loss += self.regularization()
                loss.backward()
                optimizer.step()
                running_loss += loss
            #epoch_loss = running_loss / len(batches)
            # print('Train Loss: %.6f', epoch_loss)

        self.trained = True

    def batchize(self, votes, batch_size, shuffle_rows=False):
        if shuffle_rows:
            index = np.arange(np.shape(votes)[0])
            np.random.shuffle(index)
            votes = votes[index, :]

        # Creates minibatches
        if type(votes) == sparse.coo_matrix:
            votes = votes.tocsr(votes)
        elif type(votes) == list:
            votes = sparse.csr_matrix(votes)
        batches = [(sparse.coo_matrix(
            votes[i * batch_size: (i + 1) * batch_size, :],
            copy=True),)
            for i in range(int(np.ceil(votes.shape[0] / batch_size)))
        ]

        return batches

    def cfg_verifier(self, cfg):
        try:
            if type(cfg['lr']) != float:
                raise KeyError
            if type(cfg['momentum']) != float:
                raise KeyError
            if type(cfg['step_multiplier']) != float:
                raise KeyError
            if type(cfg['seed']) != int:
                raise KeyError
            if type(cfg['epoch']) != int:
                raise KeyError
            if type(cfg['batch_size']) != int:
                raise KeyError
            if type(cfg['step_schedule']) != int:
                raise KeyError
        except KeyError:
            return False
        return True

    def weak_label(self, votes):
        """
        :param votes: (num_inst, num_lfs)
        :return: labels: (num_inst, num_classes): Soft label distribution over classes
        """


        votes = sparse.csr_matrix(votes, dtype=np.int)

        batches = self.batchize(votes, 4096, shuffle_rows=False)

        labels = np.ndarray((votes.shape[0], self.num_classes))

        offset = 0
        for votes, in batches:
            class_balance = self.class_balance - torch.logsumexp(self.class_balance, dim=0)
            lf_likelihood = self.class_conditioned_ll(votes)
            jll = class_balance + lf_likelihood
            for i in range(votes.shape[0]):
                # Iterate through instances
                p = torch.exp(jll[i, :] - torch.max(jll[i, :]))
                p = p / p.sum()
                for j in range(self.num_classes):
                    labels[offset + i, j] = p[j]
            offset += votes.shape[0]

        return labels

    def get_accuracies(self):
        """Returns the model's estimated labeling function accuracies
        :return: a NumPy array with one element in [0,1] for each labeling
                 function, representing the estimated probability that
                 the corresponding labeling function correctly outputs
                 the true class label, given that it does not abstain
        """
        acc = self.accuracy.detach().numpy()
        return np.exp(acc) / (np.exp(acc) + np.exp(-1 * acc))

    def get_propensities(self):
        """Returns the model's estimated labeling function propensities, i.e.,
        the probability that a labeling function does not abstain
        :return: a NumPy array with one element in [0,1] for each labeling
                 function, representing the estimated probability that
                 the corresponding labeling function does not abstain
        """
        prop = self.propensity.detach().numpy()
        return np.exp(prop) / (np.exp(prop) + 1)

    def get_most_probable_labels(self, votes):
        """Returns the most probable true labels given observed function outputs.
        :param votes: m x n matrix in {0, ..., k}, where m is the batch size,
                      n is the number of labeling functions and k is the number
                      of classes
        :return: 1-d Numpy array of most probable labels
        """
        return np.argmax(self.weak_label(votes), axis=1) + 1

    def get_class_balance(self):
        """Returns the model's estimated class balance
        :return: a NumPy array with one element in [0,1] for each target class,
                 representing the estimated prior probability that an example
                 has that label
        """
        return np.exp((self.class_balance - torch.logsumexp(self.class_balance, dim=0)).detach().numpy())

    def init_random(self, seed):
        """Initializes PyTorch and NumPy random seeds.

        Also sets the CuDNN back end to deterministic.

        """
        torch.backends.cudnn.deterministic = True

        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)


def get_label_distribution(label_matrix, num_classes):
    cfg = {'lr': 0.01,
           'epoch': 8,
           'seed': 0,
           'batch_size': 64,
           'momentum': 0.9,
           'step_schedule': 10,
           'step_multiplier': 1.0}

    # votes = np.random.randint(0, 3, size=(1000, 5))

    model = ClassConditionalLM(num_classes=num_classes, num_lfs=label_matrix.shape[1])

    model.optimize(label_matrix, cfg)

    return model.weak_label(label_matrix)
