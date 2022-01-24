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

    def __init__(self, config):
        """method used to define our model parameters
        Args: BarlowConfig : config = network parameters to use.
        """
        super().__init__()

        self.network_param = config.network_param
        self.optim_param = config.optim_param
        self.lr = self.optim_param.lr
        self.loss = CrossCorrelationMatrixLoss(self.network_param.lmbda)

        self.proj_dim = self.network_param.bt_proj_dim
        self.nb_proj_layers = self.network_param.nb_proj_layers
        # get backbone model and adapt it to the task
        self.backbone = get_net(
            self.network_param.backbone, self.network_param.backbone_parameters
        )
        if  self.network_param.backbone=="vit" and self.network_param.backbone_parameters is not None:
            self.patch_size = self.network_param.backbone_parameters["patch_size"]
            self.in_features = list(self.backbone.modules())[-1].in_features
            name_classif = list(self.backbone.named_children())[-1][0]
            self.backbone._modules[name_classif] = nn.Identity()
            
        elif self.network_param.backbone=="vit_dino" : # this is for the dino vit 
            self.patch_size = self.network_param.backbone_parameters["patch_size"]
            self.in_features = self.network_param.backbone_parameters["dim"]
            # self.patch_size = 16
            # self.in_features = 1000
            
        
        self.head = self.get_head()

        if self.network_param.weight_checkpoint is not None:
            pth = torch.load(self.network_param.weight_checkpoint, map_location=torch.device('cpu'))
            self.load_state_dict(pth['state_dict'], strict=True)

    def get_head(self):
        # first layer
        proj_layers = [
            nn.Linear(self.in_features, self.proj_dim, bias=False),
        ]
        for i in range(self.nb_proj_layers-1):
            proj_layers.append(nn.BatchNorm1d(self.proj_dim))
            proj_layers.append(nn.ReLU(inplace=True))
            proj_layers.append(nn.Linear(self.proj_dim, self.proj_dim, bias=False))

        return nn.Sequential(*proj_layers)

    def forward(self, x1, x2):
        # Feeding the data through the backbone and projector
        z1 = self.head(self.backbone(x1))
        z2 = self.head(self.backbone(x2))

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
        # self.optim_param.lr *= (self.trainer.datamodule.batch_size / 256) # from the paper
        optimizer = getattr(torch.optim, self.optim_param.optimizer)
        optimizer = optimizer(
            self.parameters(), lr=self.optim_param.lr, weight_decay=10e-6)
        
        if self.network_param.weight_checkpoint is not None:
            pth = torch.load(self.network_param.weight_checkpoint, map_location=torch.device('cpu'))
            optimizer.load_state_dict(pth['optimizer_states'][0])

        if self.optim_param.use_scheduler:
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_epochs=40,
                max_epochs=self.optim_param.max_epochs,
                warmup_start_lr=0.1
                * (self.optim_param.lr * self.trainer.datamodule.batch_size / 256),
                eta_min=0.1
                * (self.optim_param.lr * self.trainer.datamodule.batch_size / 256),
            )
            if self.network_param.weight_checkpoint is not None:
                pth = torch.load(self.network_param.weight_checkpoint, map_location=torch.device('cpu'))
                scheduler.load_state_dict(pth['lr_schedulers'][0])
            return [[optimizer], [scheduler]]
        else:

            return optimizer

    def _get_loss(self, batch):
        """convenience function since train/valid/test steps are similar"""
        x1, x2 = batch
        z1, z2 = self(x1, x2)
        loss = self.loss(z1, z2)

        return loss
