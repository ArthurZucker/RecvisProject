import torch.nn as nn
from models.custom_layers import BottleStack
from torchvision.models import resnet50

class Botnet50(nn.Module):
    def __init__(self, config) -> None:
        super(Botnet50, self).__init__(config)

        self.botnet = resnet50(pretrained=False)

        # replace the last layer by a BoT block as in the article
        self.botnet.layer4 = nn.Sequential(BottleStack(dim=1024, fmap_size=512))
        self.botnet.fc = nn.Linear(2048, 1024)

    def forward(self, x):
        return self.botnet(x)