import os

import torch
from PIL import Image
from torch.utils.data import Dataset


class CustomImageDataset(Dataset):
    """
    A custom dataset used to create dataloaders.
    """
    def __init__(self, filepaths, labels=None, label_map=None, transform=None):
        """
        Create a new CustomImageDataset.
        
        :param filepaths: A list of filepaths. 
        :param labels: A list of labels
        :param label_map: A dictionary to map string labels to intergers
        :param transform: A transform to perform on the images
        """
        self.filepaths = filepaths
        self.labels = labels
        self.label_map = label_map
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.filepaths[index]).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.labels is not None:
            if self.label_map is not None:
                label = torch.tensor(self.label_map[(self.labels[index])])
            else:
                label = torch.tensor(int(self.labels[index]))
            return img, label
        else:
            return img

    def __len__(self):
        return len(self.filepaths)
    

class CustomVideoDataset(Dataset):
    """
        A custom dataset used to create dataloaders.
        """
    
    def __init__(self, filepaths, labels=None, label_map=None, 
                transform_img=None, transform_vid=None, clips_dictionary=None):
        """
        Create a new CustomVideoDataset.

        :param filepaths: A list of filepaths.
        :param labels: A list of labels
        :param label_map: A dictionary to map string labels to intergers
        :param transform: A transform to perform on the frames
        :pram clips_dictionary: dictionary (id clip, list images) to get frames of a clip
        """
        self.filepaths = filepaths
        self.labels = labels
        self.label_map = label_map
        self.transform_img = transform_img
        self.transform_vid = transform_vid
        self.clips_dictionary = clips_dictionary
    
    def __getitem__(self, index):
        clip_id = int(os.path.basename(self.filepaths[index]))  # chech what path you have/want
        frames_paths = self.clips_dictionary[str(clip_id)]
        #print(f"FRAMES list[:2]: {frames_paths[:2]} and number of frames {len(frames_paths)}")
        
        frames = []
        for f in frames_paths[:10]:#[:10]:  # get same size clips - random pick for eval
            frame = Image.open(f).convert('RGB')
            #if self.transform is not None:  # BE CAREFUL TRANSFORMATION MIGHT NEED TO CHANGE FOR VIDEO EVAL!!!!!
            frame = self.transform_img(frame)
            frames.append(frame)
        
        img = torch.stack(frames)  # need to be of the same size!
        img = torch.transpose(img, 0, 1) 
        video_data = {'video':img}
        img = self.transform_vid(video_data)
        
        if self.labels is not None:
            if self.label_map is not None:
                label = torch.tensor(self.label_map[(self.labels[index])])
            else:
                label = torch.tensor(int(self.labels[index]))
            return img, label
        else:
            return img
    
    def __len__(self):
        return len(self.filepaths)


class SoftLabelDataset(Dataset):
    """
    A custom dataset used to create dataloaders.
    """
    def __init__(self, dataset, labels, remove_old_labels=False):
        """
        Create a new SoftLabelDataset.
        :param dataset: A PyTorch dataset
        :param labels: A list of labels
        :param remove_old_labels: A boolean indicating whether to the dataset returns labels that we do not use
        """
        self.dataset = dataset
        self.labels = labels
        self.remove_old_labels = remove_old_labels

    def __getitem__(self, index):
        data = self.dataset[index]
        label = self.labels[index]
        
        if self.remove_old_labels:
            data = data[0]
            
        return data, label

    def __len__(self):
        return len(self.dataset)
