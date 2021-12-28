
import numpy as np
import torch

#torch dataset library
from torch.utils.data import Dataset
from utils.transforms import BarlowTwinsTransform
from torchvision.datasets import VOCSegmentation
from PIL import Image

class BarlowTwinsDataset(VOCSegmentation):
    def __init__(self, root, img_size, image_set="trainval"):
        super().__init__(root=root, image_set=image_set)
        self.transform = BarlowTwinsTransform(img_size)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        image = np.array(image)
        # Transform the same image with 2 different transforms
        if self.transform is not None:
            aug_image1, aug_image2 = self.transform(image = image)
        return aug_image1, aug_image2    
    
