import logging
import random

import numpy as np
import PIL
import PIL.ImageOps
import PIL.ImageEnhance
import PIL.ImageDraw
from PIL import Image

from torchvision import datasets
from torchvision import transforms

logger = logging.getLogger(__name__)
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


class TransformFixMatch(object):
    def __init__(self, mean, std, input_shape, grayscale=False):
        assert len(input_shape) == 2
        header = [transforms.Grayscale(num_output_channels=3)] if grayscale else []

        self.weak = transforms.Compose(header.extend([
            transforms.Resize(input_shape),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=input_shape,
                                  padding=int(input_shape[0]*0.125),
                                  padding_mode='reflect')]))
        self.strong = transforms.Compose(header.extend([
            transforms.Resize(input_shape),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=input_shape,
                                  padding=int(input_shape[0]*0.125),
                                  padding_mode='reflect'),
            RandAugment(n=2, m=10)]))
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)


def parse_transform_compose(transform):
    s = transform.__repr__()
    l = s.split("\n")
    l = [x.replace(" ", "") for x in l]
    l = [x for x in l if "Compose" not in x]
    l = [x for x in l if x != ")"]
    return l


def is_grayscale(transform):
    try:
        parsed_t = parse_transform_compose(transform)
        for s in parsed_t:
            if "Grayscale" in s:
                return True
    except AttributeError:
        return False
