import torchmetrics
from torch.nn import functional as F


from models.base import BASE_LitModule
from models.custom_layers.unet_convs import *


class Base_Voc(BASE_LitModule):
    def __init__(self, config, bilinear=True):
        super(Base_Voc, self).__init__(config)
        self.n_channels = config.n_channels
        self.n_classes = config.n_classes
        self.bilinear = config.bilinear

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
        self.outc = OutConv(64, self.n_classes)

        # loss
        # self.loss = self.config.loss

        # metrics
        self.accuracy = torchmetrics.Accuracy()

        # save hyper-parameters to self.hparams (auto-logged by W&B)
        # requires the function to have hyper parameters __init__(self,...)
        # self.save_hyperparameters()

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
