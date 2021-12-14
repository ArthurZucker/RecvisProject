import torch.nn as nn
from torchvision.models import resnet50
from pytorch_pretrained_vit import ViT
from graphs.losses import SubcenterArcMarginProduct

class Contrastive_Vit(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.config = args
        self.net = ViT('B_16_imagenet1k', pretrained=True, image_size=384)

        
        self.num_ftrs = self.net.fc.in_features
        self.net.fc = nn.Identity()
        # self.fc =nn.Linear(
        #     num_ftrs, self.config.num_classes)
        if self.config.feature_extracting:
            for param in self.net.parameters():
                param.requires_grad = False
                

    def forward(self, x):
        x = self.net(x)
        #x = self.fc(x)
        return x
