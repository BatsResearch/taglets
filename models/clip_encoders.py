import copy
import logging
import os.path as osp

import torch
import torch.nn as nn
from clip import clip


log = logging.getLogger(__name__)


class TextEncoder(nn.Module):
    """CLIP text encoder"""

    def __init__(self, clip_model):
        super(TextEncoder, self).__init__()
        self.clip_model = clip_model

    def forward(self, text):
        encoded_text = self.clip_model.encode_text(text)
        return encoded_text


class CustomTextEncoder(torch.nn.Module):
    """This class is adapted from the codebase of "Learning to Compose Soft Prompts for Compositional Zero-Shot Learning"
    https://github.com/BatsResearch/csp/blob/main/clip_modules/text_encoder.py"""

    def __init__(self, clip_model, device, dtype):
        super(CustomTextEncoder, self).__init__()
        self.dtype = dtype
        self.clip_model = clip_model
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.token_embedding = clip_model.token_embedding
        self.device = device

    def tokenize(self, text):
        return torch.cat([clip.tokenize(tok) for tok in text])

    def forward(self, class_embeddings, classes, enable_pos_emb=True):
        """The forward function to compute representations for the prompts.

        :param class_embedding: These are the vectors for class names
        :param labels: tensor of labels (already preprocessed without _)
        :param enable_pos_emb: We set this to True since we want to account for the order

        Returns:
            torch.Tensor: the vector representation of the prompt.
        """

        prompts = [
            " ".join([" ".join(["X"] * (class_embeddings.size()[1])).strip(), c])
            for c in classes
        ]
        # log.info(f"Extended text: {prompts}")

        token_ids = clip.tokenize(prompts)

        # Get embeddings for the prompt
        text_embedding = self.token_embedding(token_ids.to(self.device))
        # for idx in range(class_embeddings.size()[0]):
        #     text_embedding[idx, 1:(class_embeddings[idx].size()[0]+1), :] = class_embeddings[idx]

        text_embedding[:, 1 : (class_embeddings[0].size()[0] + 1), :] = class_embeddings

        text_features = text_embedding.type(self.dtype)
        x = (
            text_features + self.positional_embedding.type(self.dtype)
            if enable_pos_emb
            else text_features
        )
        x = x.permute(1, 0, 2)
        # log.info(f'DEVICE: {type(self.device)}')

        if torch.cuda.is_available():
            # log.info('WRONG CPU')
            x = self.transformer(x)
        else:
            # log.info('CPU')
            x = self.transformer(x.float())
        x = x.permute(1, 0, 2)
        x = self.ln_final(x)
        tf = (
            x[torch.arange(x.shape[0]), token_ids.argmax(dim=-1)]  # POS of <EOS>
            @ self.text_projection
        )
        return tf


class ImageEncoder(nn.Module):
    """CLIP image encoder"""

    def __init__(self, clip_model):
        super(ImageEncoder, self).__init__()
        self.clip_model = clip_model

    def forward(self, text):
        encoded_image = self.clip_model.encode_image(text)
        return encoded_image


class CustomVisionTransformer(nn.Module):
    def __init__(self, vision_transformer):
        super().__init__()
        self.input_resolution = vision_transformer.input_resolution
        self.output_dim = vision_transformer.output_dim
        self.conv1 = vision_transformer.conv1

        self.class_embedding = vision_transformer.class_embedding
        self.positional_embedding = vision_transformer.positional_embedding
        self.ln_pre = vision_transformer.ln_pre

        self.transformer = vision_transformer.transformer

        self.ln_post = vision_transformer.ln_post
        self.proj = vision_transformer.proj

    def forward(
        self,
        x: torch.Tensor,
        image_prefix: torch.Tensor,
        pos_emb=True,
    ):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        #log.info(f"SHAPE AFTER CONV: {x.shape}")
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        #log.info(f"SHAPE AFTER RESHAPE: {x.shape}")
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        #log.info(f"SHAPE AFTER PERMUTE: {x.shape}")
        #log.info(f"SHAPE CLASS: {self.class_embedding.to(x.dtype).to(x.device).size()}")

        x = torch.cat(
            [
                self.class_embedding.to(x.dtype).to(x.device)
                + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                ),
                x,
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width]
        ##

        x = x + self.positional_embedding.to(x.device) if pos_emb else x
        x = self.ln_pre(x)
        log.info(f"X dtype: {x.dtype}")
        image_prefix = image_prefix.expand(x.shape[0], -1, -1)
        log.info(f"prefix dtype: {image_prefix.dtype}")
        #image_prefix = image_prefix.to(x.dtype).to(x.device) + torch.zeros(x.shape[0], image_prefix.size()[0], x.shape[-1], dtype=x.dtype, device=x.device)
        # Here we concat the prefix to the flattened patches
        x = torch.cat([
            x[:,:1,:],
            image_prefix, 
            x[:,1:,:],
        ],
        dim=1,)
        if torch.cuda.is_available():
            log.info(f"CUDAAA")
        log.info(f"after concat prefix dtype: {x.dtype}")
        log.info(f"after concat prefix shape: {x.shape}")

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x


class CustomImageEncoder(nn.Module):
    """CLIP image encoder"""

    def __init__(self, visual):
        super(CustomImageEncoder, self).__init__()
        self.visual = CustomVisionTransformer(visual)
        self.dtype = self.visual.conv1.weight.dtype

    def forward(self, image, prefix):
        encoded_image = self.visual(image.type(self.dtype), prefix.type(self.dtype))
        return encoded_image
