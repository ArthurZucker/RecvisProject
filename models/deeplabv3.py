from torchvision.models.segmentation import deeplabv3_resnet101 # deeplabv3_resnet50 does not work that well 
from models.Segmentation import Segmentation
"""
https://pytorch.org/vision/stable/models.html#torchvision.models.segmentation.deeplabv3_resnet50
"""

def Deeplabv3(config):
    return deeplabv3_resnet101(pretrained=False, num_classes=config.num_classes)
