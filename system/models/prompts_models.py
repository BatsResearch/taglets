import torch
from torch import nn

class TextPrefixModel(nn.Module):
    def __init__(self, initial_prefix, 
                 text_encoder, classes, 
                 temperature=0.07, device='cpu'):
        """ Define the model for textual prompt tuning.

        :param initial_prefix: initializes tensor of floats
        :param text_encoder: text encoder to use
        :param classes: list of classes' names
        :param temperature: fix parameter, same as clip
        :param device: device in use
        """

        super(TextPrefixModel, self).__init__()
        self.device = device
        self.initialized_prefix = initial_prefix
        self.classes = classes
        
        self.prefix = nn.Parameter(initial_prefix)
        self.text_encoder = text_encoder

    def forward(self, to_print):

        out = self.text_encoder(self.prefix, self.classes)
        norm_out = out / out.norm(dim=-1, keepdim=True)

        return out

class ImagePrefixModel(nn.Module):
    def __init__(self, initial_prefix, 
                 initial_pos_emb, image_encoder, 
                 temperature=0.07, device='cpu'):
        super(ImagePrefixModel, self).__init__()
        self.device = device
        self.initialized_prefix = initial_prefix
        
        # Initialize the model's parametwets
        self.prefix = nn.Parameter(initial_prefix) 
        self.image_pos_emb = nn.Parameter(initial_pos_emb) 
        self.image_encoder = image_encoder

    def forward(self, x):
        # Combine prefix and class embeddings to get the entire prompt representation for the
        # two augmented images
        out = self.image_encoder(x, self.prefix, self.image_pos_emb)
        norm_out = out / out.norm(dim=-1, keepdim=True)

        return out