import pickle
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class CIFARFS(Dataset):
    def __init__(self, partition, data_aug=False):
        super().__init__()
        self.partition = partition
        self.data_aug = data_aug
        
        self.img_size = (32, 32)
        # CIFAR100 stats
        self.mean = [0.5071, 0.4867, 0.4408]
        self.std = [0.2675, 0.2565, 0.2761]
        
        self.aug_transform = transforms.Compose([
            lambda x: Image.fromarray(x),
            transforms.RandomCrop(32, padding=4),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomHorizontalFlip(),
            # lambda x: np.asarray(x),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        self.noaug_transform = transforms.Compose([
            lambda x: Image.fromarray(x),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        dir = os.path.dirname(os.path.realpath(__file__))
        self.label2word = {}
        with open(os.path.join(dir, '..', 'data', 'CIFAR-FS', 'cifar-100-python', 'meta'), 'rb') as f:
            data = pickle.load(f, encoding='utf-8')
            old_label2word = data['fine_label_names']
        with open(os.path.join(dir, '..', 'data', 'CIFAR-FS',
                               f'CIFAR_FS_{partition}.pickle'), 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            self.imgs = data['data']
            labels = data['labels']
            
            # adjust sparse labels to labels from 0 to n.
            cur_class = 0
            label2label = {}
            for label in labels:
                if label not in label2label:
                    label2label[label] = cur_class
                    self.label2word[cur_class] = old_label2word[label]
                    cur_class += 1
            new_labels = []
            for label in labels:
                new_labels.append(label2label[label])
            self.labels = new_labels
    
    def get_label2word(self):
        return self.label2word
    
    def __getitem__(self, index):
        if self.data_aug:
            img = self.aug_transform(self.imgs[index])
        else:
            img = self.noaug_transform(self.imgs[index])
        label = self.labels[index]
        return img, label
    
    def __len__(self):
        return len(self.imgs)
