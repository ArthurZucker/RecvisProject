import imageio
import imgaug as ia
import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
import torch.nn as nn
from imgaug.augmentables.segmaps import SegmentationMapsOnImage


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor       
    
    def __repr__(self):
        return self.__class__.__name__ + '()'
