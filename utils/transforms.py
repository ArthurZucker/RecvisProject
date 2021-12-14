import imageio
import imgaug as ia
import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
import torch.nn as nn
from graphs.models.mobileNet import MobileNet
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
        
class SemanticSegmentation(object):
    """ Gets rid of the background and only returns a smaller image containing 
    the bird's pixels. 

    Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
    or if the numpy.ndarray has dtype = np.uint8
    In the other cases, tensors are returned without scaling.
    """

    def __init__(self, args=None):
        super().__init__()
        self.net = MobileNet()
        self.net.eval()
        self.bird_class = 3

    def __call__(self, image):
        """
                Args:
                        pic (PIL Image or numpy.ndarray): Image to be converted to tensor.
                Returns:
                        Tensor: Converted image.
        """
        
        output = self.net(image.unsqueeze(0))["out"]
        normalized_masks = torch.nn.functional.softmax(output, dim=1)
        birds_mask = (normalized_masks.argmax(dim=1) == self.bird_class)
        
        # bird_pixels = np.count_nonzero(birds_mask)
        # std = (0.229, 0.224, 0.225)
        # mean =  (0.485, 0.456, 0.406)
        # unorm = UnNormalize(mean=mean,std = std)
        # pic = np.array((unorm(image))*255, dtype=np.uint8)
        # segmap = SegmentationMapsOnImage(birds_mask[0].numpy(), shape=pic.shape)
        
        
        
        # pic = np.transpose(pic,(1,2,0))
        
        # cells = [pic,
        #     segmap.draw_on_image(pic)[0],
        #     segmap.draw(size=pic.shape[:2])[0],
        #     pic*np.repeat(np.transpose(birds_mask,(1,2,0)).numpy(),3,axis=2)
        # ]

        # # Convert cells to a grid image and save.
        # grid_image = ia.draw_grid(cells, cols=4)
        # imageio.imwrite("example_segmaps_bool.jpg", grid_image)

        return image*torch.repeat_interleave(birds_mask,3,dim=0)

    def __repr__(self):
        return self.__class__.__name__ + '()'
