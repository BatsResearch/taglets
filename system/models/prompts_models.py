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

    def forward(self):
        
        out = self.text_encoder(self.prefix, self.classes)
        norm_out = out / out.norm(dim=-1, keepdim=True)

        return out