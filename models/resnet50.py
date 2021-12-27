import torch.nn as nn
from torchvision.models import resnet50
from models.base import BASE_LitModule

# TODO implement with lightning

class Resnet50(BASE_LitModule):
    def __init__(self, config) -> None:
        super(Resnet50, self).__init__(config)
        
        self.config = config
        self.net = resnet50(pretrained=False)

    def forward(self, x):
        return self.net(x)
