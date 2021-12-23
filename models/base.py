import os

import torch
from torch.nn import Linear, CrossEntropyLoss, functional as F
from torch.optim import Adam
from pytorch_lightning import LightningModule
from kornia.losses import DiceLoss

from utils.metrics_module import MetricsModule

class BASE_LitModule(LightningModule):

    def __init__(self, config):
        '''method used to define our model parameters'''
        super().__init__()
        self.config = config
        
        # loss
        self.loss = DiceLoss()

        # optimizer parameters
        self.lr = config.lr

        # metrics
        self.metrics_module = MetricsModule(self.config.metrics)

        # save hyper-parameters to self.hparams (auto-logged by W&B)
        # self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        '''needs to return a loss from a single batch'''
        preds, loss = self._get_preds_loss_accuracy(batch)

        # Log loss and metric
        self.log('train/loss', loss)
        self.metrics_module.log_metrics("train/", self)
        return {"loss": loss, "preds": preds}

    def validation_step(self, batch, batch_idx):
        '''used for logging metrics'''
        preds, loss = self._get_preds_loss_accuracy(batch)

        # Log loss and metric
        self.log('val/loss', loss)
        self.metrics_module.log_metrics("val/", self)

        # Let's return preds to use it in a custom callback
        return preds

    def test_step(self, batch, batch_idx):
        '''used for logging metrics'''
        _, loss = self._get_preds_loss_accuracy(batch)

        # Log loss and metric
        self.log('test/loss', loss)
        self.metrics_module.log_metrics("test/", self)
    
    def configure_optimizers(self):
        '''defines model optimizer'''
        return Adam(self.parameters(), lr=self.lr)
    
    def _get_preds_loss_accuracy(self, batch):
        '''convenience function since train/valid/test steps are similar'''
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        loss = self.loss(logits, y)
        self.metrics_module.compute_metrics(preds, y)
        return preds, loss