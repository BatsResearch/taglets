import argparse
import datetime
import logging
import os
import sys
import torch
from torch.utils.data import DataLoader

from taglets.modules.multitask import MultiTaskModule
from taglets.modules.transfer import TransferModule

from .models import resnet12, resnet18, resnet24
from .datasets import CIFARFS, MiniImageNet

log = logging.getLogger(__name__)

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

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=64,
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

# def meta_test(module, test_dataset, n_ways=5, n_shots=1, num_episodes=600, n_queries=15):


def main():
    parser = argparse.ArgumentParser('argument for training')
    
    parser.add_argument('--algo', type=str, default='transfer', choices=['transfer', 'multitask'])
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
    
    args = parser.parse_args()
    
    if args.lr_decay_epochs is not None:
        iterations = args.lr_decay_epochs.split(',')
        args.lr_decay_epochs = list([])
        for it in iterations:
            args.lr_decay_epochs.append(int(it))
    
    # set up logging
    logger = logging.getLogger()
    logger.level = logging.INFO
    stream_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    model_dict = {
        'resnet12': resnet12,
        'resnet18': resnet18,
        'resnet24': resnet24,
    }
    
    if args.meta_dataset == 'CIFAR-FS':
        train_dataset = CIFARFS('train', data_aug=True)
        initial_model = model_dict[args.model](avg_pool=True,
                                               drop_rate=0.1,
                                               dropblock_size=2,
                                                  n_classes=args.n_classes)
    else:
        train_dataset = MiniImageNet('train_phase_train', data_aug=True)
        initial_model = model_dict[args.model](avg_pool=True,
                                               drop_rate=0.1,
                                               dropblock_size=5,
                                               n_classes=args.n_classes)
        
    initial_model = pre_train(initial_model, train_dataset)
    
    # TODO: Add meta_test


if __name__ == "__main__":
    main()
