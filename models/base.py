import os

import torch
from torch.nn import Linear, CrossEntropyLoss, functional as F
from torch.optim import Adam
import torchmetrics
from pytorch_lightning import LightningModule
from kornia.losses import DiceLoss
from utils.hooks import get_activation
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


    def training_step(self, batch, batch_idx):
        '''needs to return a loss from a single batch'''
        preds, loss, acc = self._get_preds_loss_accuracy(batch)

        # Log loss and metric
        self.log('train_loss', loss)
        self.log('train_accuracy', acc)

        return {"loss": loss, "preds": preds}

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
        x.requires_grad_(True)
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        loss = self.loss(logits, y)
        acc = self.accuracy(preds, y)
        return preds, loss, acc
    
    def _register_layer_hooks(self):
        self.hooks = []
        layers = self.config.layers #TODO only use those layers not every layer
        named_layers = dict(self.named_modules())
        self.features = {idx:[] for idx,i in enumerate(named_layers.keys()) if idx in layers}
        for i,k in enumerate(named_layers):
            if i in layers :
                self.hooks.append(named_layers[k].register_forward_hook(get_activation(i,self.features)))
                # named_layers[k].retain_grad()
                
    def backward(self, loss, optimizer, optimizer_idx,) -> None:
        
        loss.backward(retain_graph = True)