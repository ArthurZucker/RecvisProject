import cv2 
import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision import transforms
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
import copy


class toLongTensor(object):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, tensor):
        temp = (tensor[0] * 255).long()
        temp[temp == 255] = 0 # FIXME only for voc segmentation dataset ?
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


class SegTransform(object):
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

        image = TF.normalize(
            image, mean=self.mean, std=self.std
        )  # resizing will modify the mean and std should be place later?

        mask = toLongTensor()(mask)

        return image, mask

    def __repr__(self):
        return self.__class__.__name__ + "()"

class BarlowTwinsTransform(object):
    """TODO docstring
    """
    def __init__(self, img_size) -> None:
        
        self.img_size = img_size

    def __call__(self, image):
        
        base_transform = A.Compose(
            [A.Resize(height=self.img_size[0], width=self.img_size[1]),
            A.RandomResizedCrop(
                height=self.img_size[0],
                width =self.img_size[1],
                scale=(0.08, 1.0),
                interpolation=cv2.INTER_CUBIC,
                p=1.0,
            ),
            A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.8),
            A.HueSaturationValue(
                hue_shift_limit=int(0.1 * 180),
                sat_shift_limit=int(0.2 * 255),
                val_shift_limit=0,
                p=0.8,
            ),
            A.ToGray(p=0.2),]
        )
        transform1 = A.Compose(
            [A.Solarize(p=0.0),
            A.GaussianBlur(sigma_limit=[0.1, 0.2], p=1),
            A.HorizontalFlip(p=0.5),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
            ]
        )

        transform2 = A.Compose(
            [A.Solarize(p=0.2),
            A.GaussianBlur(sigma_limit=[0.1, 0.2], p=0.1),
            A.HorizontalFlip(p=0.5),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
            ]
        )

        base_aug_image1 = base_transform(image=image)["image"]
        base_aug_image2 = base_transform(image=image)["image"]

        aug_image1 = transform1(image=base_aug_image1)["image"]
        aug_image2 = transform2(image=base_aug_image2)["image"]

        return aug_image1, aug_image2