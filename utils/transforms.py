import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision import transforms
import random 

class toLongTensor(object):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, tensor):
        temp = (tensor[0] * 255).long()
        temp[temp == 255] = 0
        return temp

    def __repr__(self):
        return self.__class__.__name__ + "()"


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: unNormalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

    def __repr__(self):
        return self.__class__.__name__ + "()"


class transform_SS(object):
    def __init__(self, input_size, r_hflip, r_vflip, mean, std) -> None:
        super().__init__()
        self.input_size = input_size
        self.r_hflip = r_hflip
        self.r_vflip = r_vflip
        self.mean = mean
        self.std = std

    def __call__(self, image, mask):

        # Random crop
        coeff_rd_crop = np.random.uniform(0.6, 0.8)
        i, j, h, w = transforms.RandomCrop.get_params(
            image,
            output_size=(
                int(image.size[1] * coeff_rd_crop),
                int(image.size[0] * coeff_rd_crop),
            ),
        )
        image = TF.crop(image, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)

        # Resize
        resize_image = transforms.Resize(size=self.input_size)
        resize_target = transforms.Resize(
            size=self.input_size, interpolation=TF.InterpolationMode.NEAREST
        )
        image = resize_image(image)
        mask = resize_target(mask)

        # Random horizontal flipping
        if random.random() > self.r_hflip:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random vertical flipping
        if random.random() > self.r_vflip:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        # Transform to tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)

        image = TF.normalize(image, mean=self.mean, std=self.std)

        mask = toLongTensor()(mask)

        return image, mask

    def __repr__(self):
        return self.__class__.__name__ + "()"
