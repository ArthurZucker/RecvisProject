from pytorch_lightning import LightningModule
from utils.agent_utils import get_net, import_class
from utils.hooks import get_activation
import torch
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from models.unet import Unet
import models.semanticmodel
import models.resnet50
import importlib

class Segmentation(LightningModule):
    """Base semantic Segmentation class, uses the segmentation datamodule
    Supports various encoders, implements basic metric functions, and
    training steps.

    Should allow full initialization if only a backbone is provided, else if 
    a head is also given, should fuse the both of them. 

    Ex : resnet50 backbone with DeepLabV3 head. Requires a get_head function as well as the get_net 

    Args:
        LightningModule ([type]): [description]
    """

    def __init__(self, config):
        """method used to define our model parameters"""
        super().__init__()

        self.network_param = config.network_param
        self.loss_param = config.loss_param
        self.optim_param = config.optim_param

        self.rq_grad = False

        # backbone :
        # self.net = get_net(network_param.backbone, network_param)
        self.net = models.semanticmodel.SemanticModel(self.network_param)
        
        if self.network_param.backbone_parameters is not None:
            self.patch_size = self.network_param.backbone_parameters["patch_size"]
        else: 
            self.patch_size = self.net.patch_size
        
        # loss
        loss_cls = import_class(self.loss_param.name)
        self.loss = loss_cls(**self.loss_param.param)

        # optimizer parameters
        self.lr = self.optim_param.lr

    def forward(self, x):
        x.requires_grad_(self.rq_grad)
        return self.net(x)

    def training_step(self, batch, batch_idx):
        """needs to return a loss from a single batch"""
        loss, logits = self._get_preds_loss_accuracy(batch)

        # Log loss and metric
        self.log("train/loss", loss)

        return {"loss": loss, "logits": logits}

    def validation_step(self, batch, batch_idx):
        """used for logging metrics"""
        loss, logits = self._get_preds_loss_accuracy(batch)

        # Log loss and metric
        self.log("val/loss", loss)

        # Let's return preds to use it in a custom callback
        return {"logits": logits}

    def test_step(self, batch, batch_idx):
        """used for logging metrics"""
        loss, logits = self._get_preds_loss_accuracy(batch)

        # Log loss and metric
        self.log("test/loss", loss)

        return {"logits": logits}

    def predict_step(self, batch, batch_idx):
        x = batch

        logits = self(x)
        preds = torch.argmax(logits.detach(), dim=1)

        return preds

    def configure_optimizers(self):
        """defines model optimizer"""
        optimizer = getattr(torch.optim, self.optim_param.optimizer)
        optimizer = optimizer(filter(lambda p: p.requires_grad, self.parameters()), lr=self.optim_param.lr)

        # scheduler = LinearWarmupCosineAnnealingLR(
        #     optimizer,
        #     warmup_epochs=10,
        #     max_epochs=self.optim_param.max_epochs,
        #     warmup_start_lr=0.1
        #     * (self.optim_param.lr * self.trainer.datamodule.batch_size / 256),
        #     eta_min=0.1
        #     * (self.optim_param.lr * self.trainer.datamodule.batch_size / 256),
        # )  # @TODO if we need other, should be adde dbnut I doubt that will be needed

        if self.optim_param.use_scheduler :
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_epochs=5,
                max_epochs=self.optim_param.max_epochs,
                warmup_start_lr=0.1
                * (self.optim_param.lr * self.trainer.datamodule.batch_size / 256),
                eta_min=0.1
                * (self.optim_param.lr * self.trainer.datamodule.batch_size / 256),
            )
            return [[optimizer], [scheduler]]
        else:
            
            return optimizer
        # if hasattr(self.optim_param, "scheduler"):
        #     scheduler_cls = import_class(self.optim_param.scheduler)
        #     scheduler = scheduler_cls(optimizer, **self.optim_param.scheduler_parameters)
        #     lr_scheduler = {"scheduler": scheduler, "monitor": "train/loss"}
        #     return ([optimizer], [lr_scheduler])

    def backward(self, loss, optimizer, optimizer_idx) -> None:
        loss.backward(
            retain_graph=self.rq_grad
        )  # TODO only if the model is computing the erfs....

    def _get_preds_loss_accuracy(self, batch):
        """convenience function since train/valid/test steps are similar"""
        x, y = batch
        x.requires_grad_(self.rq_grad)
        logits = self(x)
        loss = self.loss(logits, y)
        return loss, logits.detach()
