import numpy as np
import torch
import torchvision.models as models


class Taglet:
    """
    Taglet class
    """
    def __init__(self):
        raise NotImplementedError()

    def execute(self, unlabeled_images, use_gpu=True):
        """
        Top: I add use_gpu as another argument for this function.
        Execute the taglet on a batch of images.
        :return: A batch of labels
        """
        raise NotImplementedError()


class ResnetTaglet(Taglet):
    def __init__(self):
        super().__init__()
        self.pretrained = True

        self.model = models.resnet18(pretrained=self.pretrained)

    def train(self, images, labels, lr=1e-3, num_epochs=100, batch_size=64, use_gpu=True):
        num_images = images.shape[0]
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        classification_criterion = torch.nn.CrossEntropyLoss()
        
        # Top: not sure if this is the most efficient way of doing it
        self.model = self.model.train()
        if use_gpu:
            self.model = self.model.cuda()
        else:
            self.model = self.model.cpu()
        
        for epoch in range(num_epochs):
            perm = torch.randperm(num_images)
            for i in range(0, num_images, batch_size):
                optimizer.zero_grad()
                
                ind = perm[i: i+batch_size]
                batch_images = images[ind]
                batch_labels = labels[ind]
                
                if use_gpu:
                    batch_images = batch_images.cuda()
                    batch_labels = batch_labels.cuda()
                
                predicted_labels = self.model(batch_images)
                loss = classification_criterion(predicted_labels, batch_labels)
                
                loss.backward()
                optimizer.step()

    def execute(self, unlabeled_images, batch_size=64, use_gpu=True):
        num_images = unlabeled_images.shape[0]
        self.model = self.model.eval()
        if use_gpu:
            self.model = self.model.cuda()
        else:
            self.model = self.model.cpu()
        list_predicted_labels = []
        for i in range(0, num_images, batch_size):
            batch_images = unlabeled_images[i: i+batch_size]
    
            if use_gpu:
                batch_images = batch_images.cuda()
    
            predicted_labels = self.model(batch_images)
            list_predicted_labels.append(predicted_labels.cpu().detach().numpy())
        all_predicted_labels = np.concatenate(list_predicted_labels)
        return all_predicted_labels


class LogisticRegressionTaglet(Taglet):
    def __init__(self):
        super().__init__()
        self.pretrained = True

        # self.model = torch.nn.Linear ...

    def train(self, images, labels, lr=1e-3, num_epochs=100, batch_size=64, use_gpu=True):
        num_images = images.shape[0]
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        classification_criterion = torch.nn.CrossEntropyLoss()
    
        # Top: not sure if this is the most efficient way of doing it
        self.model = self.model.train()
        if use_gpu:
            self.model = self.model.cuda()
        else:
            self.model = self.model.cpu()
    
        for epoch in range(num_epochs):
            perm = torch.randperm(num_images)
            for i in range(0, num_images, batch_size):
                optimizer.zero_grad()
            
                ind = perm[i: i + batch_size]
                batch_images = images[ind]
                batch_labels = labels[ind]
            
                if use_gpu:
                    batch_images = batch_images.cuda()
                    batch_labels = batch_labels.cuda()
            
                predicted_labels = self.model(batch_images)
                loss = classification_criterion(predicted_labels, batch_labels)
            
                loss.backward()
                optimizer.step()

    def execute(self, unlabeled_images, batch_size=64, use_gpu=True):
        num_images = unlabeled_images.shape[0]
        self.model = self.model.eval()
        if use_gpu:
            self.model = self.model.cuda()
        else:
            self.model = self.model.cpu()
        list_predicted_labels = []
        for i in range(0, num_images, batch_size):
            batch_images = unlabeled_images[i: i + batch_size]
        
            if use_gpu:
                batch_images = batch_images.cuda()
        
            predicted_labels = self.model(batch_images)
            list_predicted_labels.append(predicted_labels.cpu().detach().numpy())
        all_predicted_labels = np.concatenate(list_predicted_labels)
        return all_predicted_labels


class PrototypeTaglet(Taglet):
    def __init__(self):
        super().__init__()
        self.pretrained = True

        # self.model = Peilin will take care of this

    def train(self, images, labels, lr=1e-3, num_epochs=100, batch_size=64, use_gpu=True):
        num_images = images.shape[0]
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        classification_criterion = torch.nn.CrossEntropyLoss()
    
        # Top: not sure if this is the most efficient way of doing it
        self.model = self.model.train()
        if use_gpu:
            self.model = self.model.cuda()
        else:
            self.model = self.model.cpu()
    
        for epoch in range(num_epochs):
            perm = torch.randperm(num_images)
            for i in range(0, num_images, batch_size):
                optimizer.zero_grad()
            
                ind = perm[i: i + batch_size]
                batch_images = images[ind]
                batch_labels = labels[ind]
            
                if use_gpu:
                    batch_images = batch_images.cuda()
                    batch_labels = batch_labels.cuda()
            
                predicted_labels = self.model(batch_images)
                loss = classification_criterion(predicted_labels, batch_labels)
            
                loss.backward()
                optimizer.step()

    def execute(self, unlabeled_images, batch_size=64, use_gpu=True):
        num_images = unlabeled_images.shape[0]
        self.model = self.model.eval()
        if use_gpu:
            self.model = self.model.cuda()
        else:
            self.model = self.model.cpu()
        list_predicted_labels = []
        for i in range(0, num_images, batch_size):
            batch_images = unlabeled_images[i: i + batch_size]
        
            if use_gpu:
                batch_images = batch_images.cuda()
        
            predicted_labels = self.model(batch_images)
            list_predicted_labels.append(predicted_labels.cpu().detach().numpy())
        all_predicted_labels = np.concatenate(list_predicted_labels)
        return all_predicted_labels
