from kornia.losses import DiceLoss
from pytorch_lightning import LightningModule
from utils.agent_utils import import_class
from utils.hooks import get_activation
import torch

class BASE_LitModule(LightningModule):

    def __init__(self, config):
        '''method used to define our model parameters'''
        super().__init__()
        self.config = config
        self.rq_grad = False

        # loss
        name, params = next(iter(self.config.loss.items()))
        name = name.replace('_', '.')
        if "segmentation.models" in name:
            name = name.replace("segmentation.models", "segmentation_models")
        loss_cls = import_class(name)
        if "weight" in params:
            params["weight"] = torch.tensor(params["weight"])
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

    def predict_step(self, batch, batch_idx):
        x = batch
        
        logits = self(x)
        preds = torch.argmax(logits.detach(), dim=1)
        
        return preds

    def configure_optimizers(self):
        '''defines model optimizer'''
        name, params = next(iter(self.config.optimizer.items()))
        name = name.replace('_', '.')
        optimizer_cls = import_class(name)
        optimizer = optimizer_cls(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr, **params)
        
        if self.config.get("scheduler"):
            name, params = next(iter(self.config.scheduler.items()))
            name = name.replace('_', '.')
            if "lr.scheduler" in name:
                name = name.replace("lr.scheduler", "lr_scheduler")
            if "pl.bolts" in name:
                name = name.replace("pl.bolts", "pl_bolts")
            scheduler_cls = import_class(name)
            scheduler = scheduler_cls(optimizer, **params)
            lr_scheduler = {"scheduler": scheduler, "monitor": "train/iou"}
            return ([optimizer], [lr_scheduler])

        return optimizer
    
    def backward(self, loss, optimizer, optimizer_idx) -> None:
        loss.backward(retain_graph = self.rq_grad) #TODO only if the model is computing the erfs....

    def _get_preds_loss_accuracy(self, batch):
        '''convenience function since train/valid/test steps are similar'''
        x, y = batch
        x.requires_grad_(self.rq_grad)
        logits = self(x)
        loss = self.loss(logits, y)
        return loss, logits.detach()
    
    def _register_layer_hooks(self):
        self.hooks = []
        nb_erf = self.config.layers #TODO only use those layers not every layer
        named_layers = list(dict(self.named_modules())["net"].named_modules())[2:]
        layer_span = ((len(named_layers))//nb_erf) # span to take each layers
        selected_layers = named_layers[::layer_span][:nb_erf-1]+[named_layers[-1]]
        self.erf_layers_names = list(dict(selected_layers).keys())
        if nb_erf >= 2:
            self.features = {idx:[] for idx in range(len(selected_layers))} # trick to always take first and last layers FIXME later
        for i,(name, module) in enumerate(selected_layers):
            #self.hooks.append(dict(dict_layers)[name].register_forward_hook(get_activation(i, self.features)))
            self.hooks.append(module.register_forward_hook(get_activation(i, self.features)))
        
