
import numpy as np
import torch
from PIL import Image
# torch dataset library
from torch.utils.data import Dataset
from torchvision.datasets import VOCSegmentation
from utils.transforms import BarlowTwinsTransform, SegTransform


class BarlowTwinsDataset(VOCSegmentation):
    def __init__(self, root, img_size, image_set="trainval"):
        super().__init__(root=root, image_set=image_set)
        self.transform = BarlowTwinsTransform(img_size)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        image = np.array(image)
        # Transform the same image with 2 different transforms
        if self.transform is not None:
            aug_image1, aug_image2 = self.transform(image=image)
        return aug_image1, aug_image2


class BarlowTwinsDatasetSeg(VOCSegmentation):
    def __init__(self, root, img_size, image_set="trainval"):
        super().__init__(root=root, image_set=image_set)
        self.transform = SegTransform(img_size)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        target = Image.open(self.masks[idx])

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target
