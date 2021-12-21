import torch.nn as nn
from torchvision.models import resnet50
from graphs.losses import SubcenterArcMarginProduct


class Contrastive_Resnet50(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.config = args
        self.net = resnet50(pretrained=True)
        num_ftrs = self.net.fc.in_features
        self.net.fc = nn.Identity()
        self.fc = SubcenterArcMarginProduct(
            num_ftrs, self.config.num_classes, m=0.3)

        if self.config.feature_extracting:
            for param in self.net.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.net(x)
        x = self.fc(x)
        return x
