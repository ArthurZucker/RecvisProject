from pytorch_lightning import LightningModule
from utils.agent_utils import get_net, import_class
from utils.hooks import get_activation
import torch
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from .unet import Unet
import importlib

class Segmentation(LightningModule):
    """Base semantic Segmentation class, uses the segmentation datamodule
    Supports various encoders, implements basic metric functions, and
    training steps

    Args:
        LightningModule ([type]): [description]
    """

    def __init__(self, network_param, optimizer_param, loss_param):
        """method used to define our model parameters"""
        super().__init__()
        self.optimizer_param = optimizer_param
        self.rq_grad = False

        # backbone :
        # self.net = get_net(network_param.backbone, network_param)
        self.net = Unet(n_channels=network_param.n_channels,
                        n_classes=network_param.n_classes)
        # loss
        module = importlib.import_module(f"models.losses.segmentation.dice")

        self.loss = getattr(module, loss_param.name)()

        # optimizer parameters
        self.lr = optimizer_param.lr

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

        return [optimizer] #, [scheduler]]

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
