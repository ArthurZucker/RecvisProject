from pytorch_lightning import LightningModule
from utils.agent_utils import get_net, import_class
from utils.hooks import get_activation
import torch
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from models.unet import Unet
import models.deeplabv3
import models.resnet50
from utils.agent_utils import get_net,get_head

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
        self.optimizer_param = config.optim_param

        self.rq_grad = False

        if self.network_param.backbone_parameters is not None:
            self.patch_size = self.network_param.backbone_parameters["patch_size"]
            
        # intialize the backbone 
        self.backbone = get_net(
            self.network_param.backbone, self.network_param.backbone_parameters
        )
        # load weights. here state dic keys should be taken care of
        if self.network_param.weight_checkpoint is not None: 
            pth = torch.load(self.network_param.weight_checkpoint)
            state_dict = { k.replace('backbone.','') : v for k,v in pth['state_dict'].items()}
            self.backbone.load_state_dict(state_dict, strict = False)
            print(f"Loaded checkpoints from {self.network_param.weight_checkpoint}")
            if self.network_param.backbone == "vit":
                self.backbone =  Extractor(self.backbone, return_embeddings_only=True)
                
        if self.network_param.backbone_parameters is not None:
            self.patch_size = self.network_param.backbone_parameters["patch_size"]
        self.in_features = list(self.backbone.modules())[-1].in_features

        # import mlp head
        self.head = get_head(self.network_param.head,self.network_param.head_params)

        # loss
        loss_cls = import_class(self.loss_param.name)
        self.loss = loss_cls(**self.loss_param.param)

        # optimizer parameters
        self.lr = self.optimizer_param.lr

    def forward(self,x):
        x.requires_grad_(self.rq_grad)
        # @TODO @FIXME dimension will never be alright, has to correspond to the backbone> ViT should only use the extracor
        x = self.backbone(x)
        x = self.head(x)
        return x


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
        preds, loss, logits = self._get_preds_loss_accuracy(batch)

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
        optimizer = getattr(torch.optim, self.optimizer_param.optimizer)
        optimizer = optimizer(filter(lambda p: p.requires_grad, self.parameters()), lr=self.optimizer_param.lr)

        # scheduler = LinearWarmupCosineAnnealingLR(
        #     optimizer,
        #     warmup_epochs=10,
        #     max_epochs=self.optimizer_param.max_epochs,
        #     warmup_start_lr=0.1
        #     * (self.optimizer_param.lr * self.trainer.datamodule.batch_size / 256),
        #     eta_min=0.1
        #     * (self.optimizer_param.lr * self.trainer.datamodule.batch_size / 256),
        # )  # @TODO if we need other, should be adde dbnut I doubt that will be needed

        # scheduler = LinearWarmupCosineAnnealingLR(
        #     optimizer,
        #     warmup_epochs=10,
        #     max_epochs=self.optimizer_param.max_epochs,
        #     warmup_start_lr=0.1
        #     * (self.optimizer_param.lr * self.trainer.datamodule.batch_size / 256),
        #     eta_min=0.1
        #     * (self.optimizer_param.lr * self.trainer.datamodule.batch_size / 256),
        # )
        if hasattr(self.optimizer_param, "scheduler"):
            scheduler_cls = import_class(self.optimizer_param.scheduler)
            scheduler = scheduler_cls(optimizer, **self.optimizer_param.scheduler_parameters)
            lr_scheduler = {"scheduler": scheduler, "monitor": "train/loss"}
            return ([optimizer], [lr_scheduler])

        return optimizer

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
