from kornia.losses import DiceLoss
from pytorch_lightning import LightningModule
from utils.agent_utils import import_class
from utils.hooks import get_activation


class BASE_LitModule(LightningModule):

    def __init__(self, config):
        '''method used to define our model parameters'''
        super().__init__()
        self.config = config
        
        # loss
        name, params = next(iter(self.config.loss.items()))
        name = name.replace('_', '.')
        loss_cls = import_class(name)
        self.loss = loss_cls(**params)

        # optimizer parameters
        self.lr = config.lr

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
        x.requires_grad_(True)
        logits = self(x)
        loss = self.loss(logits, y)
        return loss, logits.detach()
    
    def _register_layer_hooks(self):
        self.hooks = []
        layers = self.config.layers #TODO only use those layers not every layer
        named_layers = dict(self.named_modules())
        self.features = {idx:[] for idx, i in enumerate(named_layers.keys()) if idx in layers}
        for i,k in enumerate(named_layers):
            if i in layers :
                self.hooks.append(named_layers[k].register_forward_hook(get_activation(i,self.features)))
                # named_layers[k].retain_grad()
                
    def backward(self, loss, optimizer, optimizer_idx) -> None:
        loss.backward(retain_graph = True)
        
