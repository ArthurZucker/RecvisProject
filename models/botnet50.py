import torch.nn as nn
from models.base import BASE_LitModule
from models.custom_layers import BottleStack
from torchvision.models import resnet50

# TODO implement with lightning


class Botnet50(BASE_LitModule):
    def __init__(self, config) -> None:
        super(Botnet50, self).__init__(config)
        
        self.n_classes = self.config.n_classes
        self.config = config

        self.botnet = resnet50(pretrained=False)

        # replace the last layer by a BoT block as in the article
        self.botnet.layer4 = nn.Sequential(BottleStack(dim=1024, fmap_size=512))
        self.botnet.fc = nn.Linear(2048, 1024)
        
        # TODO implement decoder head for segmentation semantic
        # self.mlp = nn.Sequential(
        #     nn.Linear(self.botnet.fc.out_features, self.botnet.fc.out_features)

        # ) 
    def forward(self, x):
        return self.botnet(x)
