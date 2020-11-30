from .module import Module
from ..data.custom_dataset import CustomDataset
from ..pipeline import Cache, Taglet
from ..scads import Scads, ScadsEmbedding

import os
import random
import torch
import logging
import numpy as np
import torchvision.transforms as transforms
import torch.nn as nn
import enum

import PIL
import PIL.ImageOps
import PIL.ImageEnhance
import PIL.ImageDraw
from PIL import Image

log = logging.getLogger(__name__)

PARAMETER_MAX = 10


# augmentations modified from https://github.com/kekmodel/FixMatch-pytorch/blob/master/dataset/randaugment.py
def auto_contrast(img, **kwarg):
    """
    input: image
    returns: an image with maximal contrast (see doc for autocontrast).
    """
    return PIL.ImageOps.autocontrast(img)


def brightness(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def color(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Color(img).enhance(v)


def contrast(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Contrast(img).enhance(v)


def cutout(img, v, max_v, bias=0):
    if v == 0:
        return img
    v = _float_parameter(v, max_v) + bias
    v = int(v * min(img.size))
    return cutout_abs(img, v)


def cutout_abs(img, v, **kwarg):
    w, h = img.size
    x0 = np.random.uniform(0, w)
    y0 = np.random.uniform(0, h)
    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = int(min(w, x0 + v))
    y1 = int(min(h, y0 + v))
    xy = (x0, y0, x1, y1)
    # gray
    color = (127, 127, 127)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img


def equalize(img, **kwarg):
    return PIL.ImageOps.equalize(img)


def identity(img, **kwarg):
    return img


def invert(img, **kwarg):
    return PIL.ImageOps.invert(img)


def posterize(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    return PIL.ImageOps.posterize(img, v)


def rotate(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.rotate(v)


def sharpness(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Sharpness(img).enhance(v)


def shear_x(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))


def shear_y(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))


def solarize(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    return PIL.ImageOps.solarize(img, 256 - v)


def solarize_add(img, v, max_v, bias=0, threshold=128):
    v = _int_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    img_np = np.array(img).astype(np.int)
    img_np = img_np + v
    img_np = np.clip(img_np, 0, 255)
    img_np = img_np.astype(np.uint8)
    img = Image.fromarray(img_np)
    return PIL.ImageOps.solarize(img, threshold)


def translate_x(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    v = int(v * img.size[0])
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def translate_y(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    v = int(v * img.size[1])
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def _float_parameter(v, max_v):
    return float(v) * max_v / PARAMETER_MAX


def _int_parameter(v, max_v):
    return int(v * max_v / PARAMETER_MAX)


def fixmatch_augment_pool():
    # FixMatch paper augmentation params (https://arxiv.org/pdf/2001.07685.pdf)
    augs = [(auto_contrast, None, None),
            (brightness, 0.9, 0.05),
            (color, 0.9, 0.05),
            (contrast, 0.9, 0.05),
            (equalize, None, None),
            (identity, None, None),
            (posterize, 4, 4),
            (rotate, 30, 0),
            (sharpness, 0.9, 0.05),
            (shear_x, 0.3, 0),
            (shear_y, 0.3, 0),
            (solarize, 256, 0),
            (translate_x, 0.3, 0),
            (translate_y, 0.3, 0)]
    return augs


class RandAugment(object):
    def __init__(self, n, m):
        assert n >= 1
        assert 1 <= m <= 10
        self.n = n
        self.m = m
        self.augment_pool = fixmatch_augment_pool()

    def __call__(self, img):
        ops = random.choices(self.augment_pool, k=self.n)
        for op, max_v, bias in ops:
            v = np.random.randint(1, self.m)
            if random.random() < 0.5:
                img = op(img, v=v, max_v=max_v, bias=bias)
        img = cutout_abs(img, int(32*0.5))
        return img


class AugmentationType(enum.Enum):
    RandAugment = 0,
    CTAugment = 1

class FixMatchModule(Module):
    def __init__(self, task):
        super().__init__(task)
        episodes = 20 if not os.environ.get("CI") else 5
        self.taglets = [FixMatchTaglet(task, verbose=True)]


# TODO: implement AMP if performance is too poor.
class FixMatchTaglet(Taglet):
    def __init__(self, task, conf_thresh=0.95, lambda_u=1,
                                    nesterov=True,
                                    mu=7,
                                    weight_decay=5e-4,
                                    temp=1,
                                    aug_type=AugmentationType.RandAugment,
                                    use_amp=False,
                                    verbose=False):
        self.conf_thresh = conf_thresh
        self.lambda_u = lambda_u
        self.nesterov = nesterov
        self.mu = mu
        self.weight_decay = weight_decay
        self.temp = temp
        self.aug_type = aug_type
        self.use_amp = use_amp

        if verbose:
            log.info('Initializing FixMatch with hyperparameters:')
            log.info('confidence threshold: %.4f', self.conf_thresh)
            log.info('nesterov: ' + str(self.nesterov))
            log.info("unlabeled loss weight (lambda u): %.4f", self.lambda_u)
            log.info('temperature: %.4f', self.temp)
            log.info('augmentation type: ' + str(self.aug_type))
            log.info('use mixed precision: ' + str(self.use_amp))

        if aug_type is AugmentationType.CTAugment:
            log.warning('CTAugment has not been implemented yet. Defaulting to RandAugment.')
            self.aug_type = AugmentationType.RandAugment

        super().__init__(task)

