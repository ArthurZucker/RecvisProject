import torch.nn as nn
from torchvision.models import resnet50

class Resnet50(nn.Module):
    def __init__(self, pretrained=False) -> None:
        super().__init__()
        self.net = resnet50(pretrained=pretrained)

    def forward(self, x):
        return self.net(x)
