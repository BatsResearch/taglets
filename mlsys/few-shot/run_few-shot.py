import argparse
import datetime
import logging
import numpy as np
import os
import random
import scipy
import sys
import torch
from torch.utils.data import DataLoader, TensorDataset

from taglets.modules.multitask import MultiTaskModule
from taglets.modules.transfer import TransferModule
from taglets.task import Task
from taglets.scads import Scads
from taglets.task.utils import labels_to_concept_ids

from .models import resnet12, resnet18, resnet24
from .datasets import CIFARFS, MiniImageNet

log = logging.getLogger(__name__)


# Reference: https://github.com/WangYueFt/rfs/blob/master/eval/meta_eval.py
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t._ppf((1 + confidence) / 2., n - 1)
    return m, h


def pre_train(model, train_dataset):
    # ------------------training params---------------------
    lr = 0.05
    momentum = 0.9
    weight_decay = 5e-4
    lr_decay_epochs = [60, 80]
    lr_decay_rate = 0.1
    num_epochs = 100
    save_freq = 10
    batch_size = 64
    # ------------------------------------------------------

    save_dir = os.path.join(os.getcwd(), 'saved_models', datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  drop_last=True)
    
    
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_decay_epochs, gamma=lr_decay_rate)
    criterion = torch.nn.CrossEntropyLoss()
    
    model = model.cuda()

    best_ep = 0
    best_model = None
    for ep in range(num_epochs):
        log.info('Epoch: {}'.format(ep))

        running_loss = 0
        running_acc = 0
        dataset_size = 0
        for batch in train_dataloader:
            inputs = batch[0]
            labels = batch[1]
            inputs = inputs.cuda()
            labels = labels.cuda()
    
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                preds = torch.argmax(outputs, dim=1)
                loss.backward()
                optimizer.step()
    
            running_loss += loss.item() * inputs.size(0)
            running_acc += torch.sum(preds == labels).item()
            dataset_size += inputs.size(0)

        train_loss = running_loss / dataset_size
        train_acc = running_acc / dataset_size
        
        scheduler.step()
        log.info('train loss: {:.4f}'.format(train_loss))
        log.info('train acc: {:.4f}%'.format(train_acc * 100))
    
        if ep % save_freq == 0 and save_dir is not None:
            log.info('==> Saving model at epoch {}...'.format(ep))
            state = {
                'epoch': ep,
                'model': model.state_dict(),
            }
            save_file = os.path.join(save_dir, f'model_epoch_{ep}.pth')
            torch.save(state, save_file)

    if save_dir is not None:
        # save the last model
        log.info('==> Saving last model...')
        state = {
            'epoch': num_epochs,
            'model': model.state_dict(),
        }
        save_file = os.path.join(save_dir, f'model_last.pth')
        torch.save(state, save_file)
    
        if best_model is not None:
            log.info(f'==> Saving best model from epoch {best_ep}...')
            state = {
                'epoch': best_ep,
                'model': best_model.state_dict(),
            }
            save_file = os.path.join(save_dir, f'model_best.pth')
            torch.save(state, save_file)
    return model


def meta_test(initial_model, training_module, test_dataset, n_ways=5, n_shots=1, num_episodes=600, n_queries=15, **kwargs):
    data_by_class = {}
    for idx in range(len(test_dataset.imgs)):
        if test_dataset.labels[idx] not in data_by_class:
            data_by_class[test_dataset.labels[idx]] = []
        data_by_class[test_dataset.labels[idx]].append(test_dataset.imgs[idx])
    classes = list(data_by_class.keys())
    n_classes = len(classes)
    
    initial_model.fc = torch.nn.Identity()
    
    acc_list = []
    for episode in range(num_episodes):
        log.info(f'Episode: {episode}')
        
        # ---------------------- Prepare data for the episode ----------------------------
        # Fix random seed
        np.random.seed(episode)
        random.seed(episode)

        # Sample classes
        done = False
        while not done:
            done = True
            cls_sampled = np.random.choice(n_classes, n_ways, False)
            for cls in cls_sampled:
                if n_shots + n_queries > len(data_by_class[cls]):
                    done = False
                    break

        support_xs = []
        support_ys = []
        query_xs = []
        query_ys = []
        for idx, cls in enumerate(cls_sampled):
            imgs = np.asarray(data_by_class[cls]).astype('uint8')
            support_xs_ids_sampled = np.random.choice(range(imgs.shape[0]), n_shots, False)
            support_xs.append(imgs[support_xs_ids_sampled])
            support_ys.append([idx] * n_shots)
            query_xs_ids = np.setxor1d(np.arange(imgs.shape[0]), support_xs_ids_sampled)
            query_xs_ids = np.random.choice(query_xs_ids, n_queries, False)
            query_xs.append(imgs[query_xs_ids])
            query_ys.append([idx] * query_xs_ids.shape[0])
        channel, height, width = query_xs[0][0].shape
        support_xs = torch.FloatTensor(support_xs).reshape((-1, channel, height, width))
        query_xs = torch.FloatTensor(query_xs).reshape((-1, channel, height, width))
        support_ys = torch.LongTensor(support_ys).reshape(-1)
        query_ys = torch.LongTensor(query_ys).reshape(-1)
        
        episode_train_dataset = TensorDataset(support_xs, support_ys)
        episode_test_dataset = TensorDataset(query_xs, query_ys)
        # ------------------------------------------------------------------------------------
        
        task = Task('name',
                    labels_to_concept_ids(classes),
                    (height, width),
                    episode_train_dataset,
                    None,
                    None,
                    scads_path='./predefined/scads.imagenet22k.sqlite3',
                    scads_embedding_path='./predefined/embeddings/numberbatch-en19.08.txt.gz')
        task.set_initial_model(initial_model)
        
        taglet_module = training_module(task)
        taglet_module.train_taglets(episode_train_dataset, None, None)
        taglet = taglet_module.get_valid_taglets()
        
        acc = taglet.evaluate(episode_test_dataset)
        acc_list.append(acc)
        m, h = mean_confidence_interval(acc)
        
        log.info('episode acc: {:.4f}'.format(acc))
        log.info('aggregate 95% interval acc: {} +/- {}'.format(m, h))
    return acc_list

def main():
    parser = argparse.ArgumentParser('argument for training')
    
    parser.add_argument('--module', type=str, default='multitask', choices=['multitask', 'transfer'])
    parser.add_argument('--dataset', type=str, default='miniImageNet', choices=['miniImageNet',
                                                                                'CIFAR-FS',
                                                                                'tieredImageNet',
                                                                                'FC100'])
    
    # few-shot params
    parser.add_argument('--n_classes', type=int, default=64, help='Number of base classes')
    parser.add_argument('--n_ways', type=int, default=5, help='Number of ways in meta testing phase')
    parser.add_argument('--num_episodes', type=int, default=600, help='Number of test runs')
    parser.add_argument('--n_shots', type=int, default=1, help='Number of shots in test')
    parser.add_argument('--n_queries', type=int, default=15, help='Number of query in test')
    
    # model
    parser.add_argument('--model', type=str, default='resnet12', choices=['resnet12', 'resnet18', 'resnet24'])
    parser.add_argument('--model_path', type=str, help='path to pre-trained model')
    
    parser.add_argument('--scads_root_path', type=str, default='/users/wpiriyak/data/bats/datasets')
    
    args = parser.parse_args()
    
    model_dict = {
        'resnet12': resnet12,
        'resnet18': resnet18,
        'resnet24': resnet24,
    }
    
    module_dict = {
        'transfer': TransferModule,
        'multitask': MultiTaskModule
    }
    
    if args.dataset == 'CIFAR-FS':
        train_dataset = CIFARFS('train', data_aug=True)
        test_dataset = CIFARFS('test')
        initial_model = model_dict[args.model](avg_pool=True,
                                               drop_rate=0.1,
                                               dropblock_size=2,
                                                  n_classes=args.n_classes)
    else:
        train_dataset = MiniImageNet('train_phase_train', data_aug=True)
        test_dataset = MiniImageNet('test')
        initial_model = model_dict[args.model](avg_pool=True,
                                               drop_rate=0.1,
                                               dropblock_size=5,
                                               n_classes=args.n_classes)

    if args.model_path is not None:
        save_params = torch.load(args.model_path)
        initial_model.load_state_dict(save_params['model'])
    else:
        initial_model = pre_train(initial_model, train_dataset)

    Scads.set_root_path(args.scads_root_path)

    kwargs = {k: v for k, v in vars(args).items() if v is not None}
    acc_list = meta_test(initial_model, module_dict[args.module], test_dataset, **kwargs)

    log.info(f'all results - {acc_list}')

if __name__ == "__main__":
    main()
