from pytorch_lightning import LightningModule
from kornia.losses import DiceLoss
from utils.agent_utils import import_class

class BASE_LitModule(LightningModule):

    def __init__(self, config):
        '''method used to define our model parameters'''
        super().__init__()
        self.config = config
        
        # loss
        self.loss = DiceLoss()

        # optimizer parameters
        self.lr = config.lr

        # save hyper-parameters to self.hparams (auto-logged by W&B)
        # self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        '''needs to return a loss from a single batch'''
        loss, logits = self._get_preds_loss_accuracy(batch)

        # Log loss and metric
        self.log('train/loss', loss)

        return {"loss": loss, "logits": logits}

    def validation_step(self, batch, batch_idx):
        '''used for logging metrics'''
        loss, logits = self._get_preds_loss_accuracy(batch)

        # Log loss and metric
        self.log('val/loss', loss)

        # Let's return preds to use it in a custom callback
        return {"logits": logits}

    def test_step(self, batch, batch_idx):
        '''used for logging metrics'''
        preds, loss, logits = self._get_preds_loss_accuracy(batch)

        # Log loss and metric
        self.log('test/loss', loss)
        
        return {"logits": logits}

    def configure_optimizers(self):
        '''defines model optimizer'''
        name, params = next(iter(self.config.optimizer.items()))
        name = name.replace('_', '.')
        optimizer_cls = import_class(name)
        optimizer = optimizer_cls(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr, **params)
        return optimizer

    def _get_preds_loss_accuracy(self, batch):
        '''convenience function since train/valid/test steps are similar'''
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        return loss, logits.detach()