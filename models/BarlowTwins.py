import torch.nn as nn
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pytorch_lightning import LightningModule
from torch.nn import functional as F
import torch
from utils.agent_utils import get_net

from models.losses.barlow_twins import CrossCorrelationMatrixLoss


class BarlowTwins(LightningModule):
    """General class for self supervised learning using BarlowTwins method
    The encoder used as backbone can be defined in the hparams file
    """
    def __init__(self, network_param,optim_param = None):
        """method used to define our model parameters
        Args: BarlowConfig : config = network parameters to use. 
        """
        super().__init__()
        self.loss = CrossCorrelationMatrixLoss(network_param.lmbda)
        self.optim_param = optim_param
        self.proj_dim = network_param.bt_proj_dim
        # get backbone model and adapt it to the task
        self.encoder = get_net(network_param.encoder,network_param) 
        self.in_features = list(self.encoder.modules())[-1].in_features
        name_classif = list(self.encoder.named_children())[-1][0]
        self.encoder._modules[name_classif] = nn.Identity()
        self.head = self.get_head()
        
    
    def get_head(self):
        # first layer
        proj_layers = [nn.Linear(self.in_features, self.proj_dim, bias=False), nn.GELU()]
        for i in range(self.nb_proj_layers):
            proj_layers.append(nn.BatchNorm1d(self.proj_dim))
            proj_layers.append(nn.ReLU(inplace=True))
            proj_layers.append(nn.Linear(self.proj_dim, self.proj_dim, bias=False))
            
        return nn.Sequential(*proj_layers)


    def forward(self, x1, x2):
        # Feeding the data through the encoder and projector
        z1 = self.proj(self.encoder(x1))
        z2 = self.proj(self.encoder(x2))

        return z1, z2

    def training_step(self, batch, batch_idx):
        """needs to return a loss from a single batch"""
        loss = self._get_loss(batch)
        self.log("train/loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        """used for logging metrics"""
        loss = self._get_loss(batch)
        self.log("val/loss", loss)

        return loss

    def configure_optimizers(self):
        """defines model optimizer"""
        optimizer = getattr(torch.optim,self.optim_param.optimizer)
        optimizer = optimizer(self.parameters(), lr=self.optim_param.lr)
        # scheduler = LinearWarmupCosineAnnealingLR(
        #     optimizer, warmup_epochs=5, max_epochs=40
        # )
        return optimizer #[[optimizer], [scheduler]]


    def _get_loss(self, batch):
        """convenience function since train/valid/test steps are similar"""
        x1, x2 = batch
        z1, z2 = self(x1, x2)
        loss = self.loss(z1, z2)

        return loss

