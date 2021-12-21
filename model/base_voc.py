import os

import torch
import torchmetrics
from pytorch_lightning import LightningModule
from torch.nn import CrossEntropyLoss, Linear
from torch.nn import functional as F
from torch.optim import Adam

from model.base import BASE_LitModule
from model.custom_layers.unet_convs import *

class VOC_LitModule(BASE_LitModule):
    
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(VOC_LitModule, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

        # loss
        self.loss = self.config.loss

        # optimizer parameters
        self.lr = self.config.lr

        # metrics
        self.accuracy = torchmetrics.Accuracy()

        # save hyper-parameters to self.hparams (auto-logged by W&B)
        self.save_hyperparameters()

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
        logits = self.outc(x)
        return logits