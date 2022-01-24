from pytorch_lightning import LightningModule
from utils.agent_utils import get_net, import_class
from utils.hooks import get_activation
import torch
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
import models.semanticmodel
import models.resnet50
from utils.agent_utils import get_net, get_head
from easydict import EasyDict
from vit_pytorch.extractor import Extractor


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

        self.input_size = config.data_param.input_size
        self.rq_grad = False
        self.max_epochs = config.hparams.max_epochs
        if self.network_param.backbone_parameters is not None:
            self.patch_size = self.network_param.backbone_parameters["patch_size"]

        # intialize the backbone
        self.backbone = get_net(
            self.network_param.backbone, self.network_param.backbone_parameters
        )
        # load weights. here state dic keys should be taken care of
        if self.network_param.backbone_checkpoint is not None:
            pth = torch.load(self.network_param.backbone_checkpoint)
            state_dict = {k.replace('backbone.', ''): v for k, v in pth['state_dict'].items()}
            self.backbone.load_state_dict(state_dict, strict=False)
            print(
                f"Loaded checkpoints from {self.network_param.backbone_checkpoint}")
        if self.network_param.backbone == "vit":
            self.backbone = Extractor(
                self.backbone, return_embeddings_only=True)

        # backbone :
        # self.net = get_net(network_param.backbone, network_param)
        self.net = models.semanticmodel.SemanticModel(self.network_param)
        
        if "vit" in self.network_param.backbone :
            if self.network_param.backbone_parameters is not None:
                self.patch_size = self.network_param.backbone_parameters["patch_size"]
            else: 
                self.patch_size = self.net.patch_size
        
        if self.network_param.backbone_parameters is not None:
            self.patch_size = self.network_param.backbone_parameters["patch_size"]
        self.in_features = list(self.backbone.modules())[-1].in_features
        out_features = list(self.backbone.modules())[-1].out_features
        
        self.network_param.head_params = EasyDict({"input_dim": self.in_features, "img_size": self.input_size,
                                          "patch_size": self.patch_size,"decoder_hidden_size":768,
                                          "num_labels":21})
        # import mlp head
        self.head = get_head(self.network_param.head,
                             self.network_param.head_params)

        # loss
        loss_cls = import_class(self.loss_param.name)
        self.loss = loss_cls(**self.loss_param.param)

        # optimizer parameters
        self.lr = self.optim_param.lr

    def forward(self, x):
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
