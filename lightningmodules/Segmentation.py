from pytorch_lightning import LightningModule
from utils.agent_utils import get_net, import_class
from utils.hooks import get_activation
import torch
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from models.unet import Unet
import models.deeplabv3
import models.resnet50
import importlib
from torchvision.models import resnet50

class Segmentation(LightningModule):
    """Base semantic Segmentation class, uses the segmentation datamodule
    Supports various encoders, implements basic metric functions, and
    training steps

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

        # backbone :
        # self.net = get_net(network_param.backbone, network_param)
        # self.net = Unet(n_channels=network_param.n_channels,
        #                 n_classes=network_param.n_classes)
        # encoder = models.resnet50.Resnet50()
        if self.network_param.model == "deeplabv3":
            self.net = models.deeplabv3.Deeplabv3(
                num_classes=self.network_param.n_classes, encoder=self.network_param.encoder)
        else:
            raise ValueError(f'option {self.network_param.model} does not exist !')

        # loss
        module = importlib.import_module(f"models.losses.segmentation.dice")
        self.loss = getattr(module, self.loss_param.name)()

        # optimizer parameters
        self.lr = self.optimizer_param.lr

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
        optimizer = optimizer(self.parameters(), lr=self.optimizer_param.lr)

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

        return [optimizer]  # , [scheduler]]

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
