from models.custom_layers.unet_convs import *
from utils.hooks import get_activation
import torch.nn as nn
"""
https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
"""

class Unet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super().__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(self.n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.fc = OutConv(64, self.n_classes)

        # checkpoint = 'https://github.com/milesial/Pytorch-UNet/releases/download/v2.0/unet_carvana_scale0.5_epoch1.pth'
        # self.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint, progress=True), strict=False, map_location=torch.device('cpu'))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.fc(x)
        return logits
