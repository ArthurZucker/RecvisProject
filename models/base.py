import os

import torch
from torch.nn import Linear, CrossEntropyLoss, functional as F
from torch.optim import Adam
import torchmetrics
from pytorch_lightning import LightningModule
from kornia.losses import DiceLoss

class BASE_LitModule(LightningModule):

    def __init__(self,config):
        '''method used to define our model parameters'''
        super().__init__()
        # loss
        self.loss = DiceLoss()

        # optimizer parameters
        self.lr = config.lr

        # metrics
        self.accuracy = torchmetrics.Accuracy()

        # save hyper-parameters to self.hparams (auto-logged by W&B)
        # self.save_hyperparameters()

    # def forward(self, x):
    #     '''method used for inference input -> output'''

    #     batch_size, channels, width, height = x.size()

    #     # (b, 1, 28, 28) -> (b, 1*28*28)
    #     x = x.view(batch_size, -1)

    #     # let's do 3 x (linear + relu)
    #     x = self.layer_1(x)
    #     x = F.relu(x)
    #     x = self.layer_2(x)
    #     x = F.relu(x)
    #     x = self.layer_3(x)

    #     return x

    def training_step(self, batch, batch_idx):
        '''needs to return a loss from a single batch'''
        _, loss, acc = self._get_preds_loss_accuracy(batch)

        # Log loss and metric
        self.log('train_loss', loss)
        self.log('train_accuracy', acc)

        return loss

    def validation_step(self, batch, batch_idx):
        '''used for logging metrics'''
        preds, loss, acc = self._get_preds_loss_accuracy(batch)

        # Log loss and metric
        self.log('val_loss', loss)
        self.log('val_accuracy', acc)

        # Let's return preds to use it in a custom callback
        return preds

    def test_step(self, batch, batch_idx):
        '''used for logging metrics'''
        _, loss, acc = self._get_preds_loss_accuracy(batch)

        # Log loss and metric
        self.log('test_loss', loss)
        self.log('test_accuracy', acc)
    
    def configure_optimizers(self):
        '''defines model optimizer'''
        return Adam(self.parameters(), lr=self.lr)
    
    def _get_preds_loss_accuracy(self, batch):
        '''convenience function since train/valid/test steps are similar'''
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        loss = self.loss(logits, y)
        acc = self.accuracy(preds, y)
        return preds, loss, acc