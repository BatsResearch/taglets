import pickle
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from nltk.corpus import wordnet as wn


def wnid_to_name(wnid):
    synset = wn.synset_from_pos_and_offset('n', int(wnid[1:]))
    return synset.lemmas()[0].name()


class MiniImageNet(Dataset):
    def __init__(self, partition, data_aug=False):
        super().__init__()
        self.partition = partition
        self.data_aug = data_aug
        
        self.img_size = (84, 84)
        # ImageNet stats
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        
        self.aug_transform = transforms.Compose([
            lambda x: Image.fromarray(x),
            transforms.RandomCrop(84, padding=8),
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
        dir = os.getcwd()
        with open(os.path.join(dir, 'data', 'miniImageNet',
                               f'miniImageNet_category_split_{partition}.pickle'), 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            self.imgs = data['data']
            
            labels = data['labels']
            # adjust sparse labels to labels from 0 to n.
            cur_class = 0
            label2label = {}
            for label in labels:
                if label not in label2label:
                    label2label[label] = cur_class
                    cur_class += 1
            new_labels = []
            for label in labels:
                new_labels.append(label2label[label])
            self.labels = new_labels
            
            catname2label = data['catname2label']
            self.label2wnid = {label2label[v]: k for k, v in catname2label.items()}
            self.label2word = {label2label[v]: wnid_to_name(k) for k, v in catname2label.items()}
    
    def get_label2wnid(self):
        return self.label2wnid
    
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
