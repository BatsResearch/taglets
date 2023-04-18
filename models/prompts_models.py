import logging

import torch
from torch import nn

log = logging.getLogger(__name__)


class TextPrefixModel(nn.Module):
    def __init__(
        self, initial_prefix, text_encoder, classes, temperature=0.07, device="cpu"
    ):
        """Define the model for textual prompt tuning.

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

    def forward(self, classes):
        # log.info(f"classes: {classes}")
        out = self.text_encoder(self.prefix, classes)
        norm_out = out / out.norm(dim=-1, keepdim=True)

        return out


class ImagePrefixModel(nn.Module):
    def __init__(
        self,
        initial_prefix,
        image_encoder,
        temperature=0.07,
        device="cpu",
    ):
        super(ImagePrefixModel, self).__init__()
        self.device = device
        self.initialized_prefix = initial_prefix

        # Initialize the model's parametwets
        self.prefix = nn.Parameter(initial_prefix)
        self.image_encoder = image_encoder

    def forward(self, x):
        # Combine prefix and class embeddings to get the entire prompt representation for the
        # two augmented images
        out = self.image_encoder(x, self.prefix)
        norm_out = out / out.norm(dim=-1, keepdim=True)

        return out


class UPTModel(nn.Module):
    def __init__(
        self,
        coop_embeddings,
        vpt_embeddings,
        vpt_embeddings_deep,
        linear_layers,
        transformer,
        image_encoder,
        text_encoder,
        classes,
        temperature=0.07,
        device="cpu",
        dtype=torch.float32,
    ):
        super(UPTModel, self).__init__()
        self.device = device
        self.classes = classes
        self.temperature = temperature
        self.dtype = dtype

        # Initialize the model's parameters
        self.coop_embeddings = nn.Parameter(coop_embeddings)
        self.vpt_embeddings = nn.Parameter(vpt_embeddings)

        self.coop_length = self.coop_embeddings.size()[1]
        self.coop_dim = self.coop_embeddings.size()[2]

        self.vpt_length = self.vpt_embeddings.size()[1]
        self.vpt_dim = self.vpt_embeddings.size()[2]

        if vpt_embeddings_deep is not None:
            self.vpt_embeddings_deep = nn.Parameter(vpt_embeddings_deep)
        else:
            self.vpt_embeddings_deep = None

        self.proj_coop_pre = linear_layers[0]
        self.proj_coop_post = linear_layers[1]
        self.proj_vpt_pre = linear_layers[2]
        self.proj_vpt_post = linear_layers[3]
        self.transformer = transformer

        self.image_encoder = image_encoder
        self.text_encoder = text_encoder

    # Given coop_embeddings, vpt_embeddings, and vpt_embeddings_deep
    # - Project into 128 dim space
    # - Run sequence through transformer
    # - Project back to CLIP (512) dim space
    # (Error when there is no input arg. https://github.com/pytorch/pytorch/pull/37902)
    def forward(self, dummy_input):
        # First, we project the prompts into lower dim space, and concat them, and make them correct dtype
        coop_embeddings = self.coop_embeddings + dummy_input
        coop_embds = self.proj_coop_pre(coop_embeddings).to(self.device)
        if self.vpt_embeddings_deep is not None:
            vpt_embds = torch.cat((self.vpt_embeddings, self.vpt_embeddings_deep), dim=0).to(self.device)
        vpt_embds = self.proj_vpt_pre(self.vpt_embeddings).to(self.device)
        # vpt_embds = vpt_embds.reshape((-1, self.vpt_length, self.vpt_dim)) #flatten if they are deep embds
        # concat coop and vpt prompts
        prompt_seq = torch.cat((coop_embds, vpt_embds), dim=0).to(torch.float32) # TODO: Fix hacky type change
        
        # Then, we run the sequence through the transformer
        output_seq = self.transformer(prompt_seq).to(torch.float16) # TODO: Fix hacky type change
        
        # Finally, we project the seq back into prompt space
        coop_embs = self.proj_coop_post(output_seq[:len(self.coop_embeddings)].to(self.dtype)).reshape(-1, self.coop_length, self.coop_dim)
        vpt_embs = self.proj_vpt_post(output_seq[len(self.coop_embeddings):].to(self.dtype)).reshape(-1, self.vpt_length, self.vpt_dim)
        vpt_emb_deep = None if vpt_embs.shape[0] == 1 else vpt_embs[1:, :, :]
        
        return coop_embs, vpt_embs, vpt_emb_deep