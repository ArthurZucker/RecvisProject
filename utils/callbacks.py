
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import wandb
from pytorch_lightning.callbacks import Callback
from torch.autograd import grad

from utils.constant import PASCAL_VOC_classes
from utils.metrics_module import MetricsModule


class LogPredictionsCallback(Callback):
    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Called when the validation batch ends."""

        # `outputs` comes from `LightningModule.validation_step`
        # which corresponds to our model predictions in this case

        # Let's log 20 sample image predictions from first batch
        if batch_idx == 0:
            self.log_images("validation", batch, 5, outputs)

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Called when the training batch ends."""

        # `outputs` comes from `LightningModule.validation_step`
        # which corresponds to our model predictions in this case

        # Let's log 20 sample image predictions from first batch
        if batch_idx == 0:
            self.log_images("train", batch, 5, outputs)

    def log_images(self, name, batch, n, outputs):

        x, y = batch
        images = x[:n].cpu()
        ground_truth = np.array(y[:n].cpu())

        logits = outputs["logits"]  # preds
        preds = torch.argmax(logits, dim=1)

        predictions = np.array(preds[:n].cpu())

        samples = []

        mean = np.array([0.485, 0.456, 0.406])  # TODO this is not beautiful
        std = np.array([0.229, 0.224, 0.225])

        for i in range(images.shape[0]):

            bg_image = images[i].detach().numpy().transpose((1, 2, 0))
            bg_image = std * bg_image + mean
            bg_image = np.clip(bg_image, 0, 1)

            prediction_mask = predictions[i]
            true_mask = ground_truth[i]

            samples.append(
                wandb.Image(
                    bg_image,
                    masks={
                        "prediction": {
                            "mask_data": prediction_mask,
                            "class_labels": PASCAL_VOC_classes,
                        },
                        "ground truth": {
                            "mask_data": true_mask,
                            "class_labels": PASCAL_VOC_classes,
                        },
                    },
                )
            )
        wandb.log({name: samples})


class LogMetricsCallback(Callback):
    def __init__(self, config):
        self.config = config

    def on_fit_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        device = pl_module.device

        self.metrics_module_train = MetricsModule(
            "train", self.config.metrics, device, self.config.n_classes
        )

        self.metrics_module_validation = MetricsModule(
            "val", self.config.metrics, device
        )

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Called when the train batch ends."""

        _, y = batch
        self.metrics_module_train.update_metrics(outputs["logits"], y)

    def on_train_epoch_end(self, trainer, pl_module):
        """Called when the train epoch ends."""

        self.metrics_module_train.log_metrics("train/", pl_module)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Called when the validation batch ends."""

        _, y = batch
        self.metrics_module_validation.update_metrics(outputs["logits"], y)

    def on_validation_epoch_end(self, trainer, pl_module):
        """Called when the validation epoch ends."""

        self.metrics_module_validation.log_metrics("val/", pl_module)


class LogERFVisualizationCallback(Callback):
    # TODO add feature vizualization for training and validation data
    # should probably use hooks to keep the model structure
    # classe image = classe predominante dans l'image + prendre features encoder
    # We shall use the activation map in order to visualize the effective receptive fields
    # those can be obtained by projecting the activation maps from upper layers
    # and resizing with a deconvolution or with interpolation ?
    # the original paper uses an average over 1000 images, this can also be considered.

    # implementation will rely on the only git available out there : https://github.com/mcFloskel/houseNet/wiki/Effective-receptive-field
    # and the original paper giving the definition of the effective receptive field
    # It is defined as the partial derivative of the center pixel of the output map
    # with respect to the input map (we can set the input as an input image and take the various output maps

    def __init__(self, config) -> None:
        """Initialize the callback with the layers
        to use to compute the effective receptive fields
        FOr now TODO define the format (most probably ints for the stage or the index of the layer)

        Args:
            layers ([type]): [description]
        """
        super().__init__()
        self.config = config
        self.layers = config.layers
        self.eps = 1e-7
        self.gradient = {i: self.eps for i in range(self.layers)}

    # from different stages of the network)
    # Our implementation should be network independant
    # def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:
    #     if (pl_module.current_epoch%self.config.erf_freq == 0 and batch_idx == 0):
    #         pl_module._register_layer_hooks()
    #         pl_module.rq_grad = True

    def on_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if (pl_module.current_epoch) % self.config.erf_freq == 0:
            pl_module._register_layer_hooks()
            pl_module.rq_grad = True

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        # hooks have already been placed to extract the output
        # here, compute the gradient of the various ouputs with respect
        # to the input feature map, average it over the validation batches
        # until we have the number of images we required
        if not trainer.sanity_checking and pl_module.current_epoch % self.config.erf_freq == 0: # exclude last batch were batch norm is appliedss FIXME
            self.hooks = pl_module.features
            x, _ = batch
            for name in range(self.layers):
                # for each layer, compute the mean
                gradient_wrt_ipt = self.hooks[name]
                if gradient_wrt_ipt != []:
                    try:
                        gradient_wrt_ipt = grad(gradient_wrt_ipt, x, retain_graph=True)[
                            0
                        ].detach()
                        # TODO check whether the abs should be done before the mean or after
                        self.gradient[name] += (np.squeeze(torch.mean(torch.sum(torch.abs(gradient_wrt_ipt), axis= 1),axis=0).cpu().numpy())-self.gradient[name])/(batch_idx+1)
                            # average over the batches but sum over the channels 

                        del self.hooks[name]
                        self.hooks[name] = []# reset the hooks for the batch
                    except Exception as e:
                        # the gradient can't be computed because of batchnorm, jsut ignore
                        print(f"Tried to compute gradient error : {e}, cleaning up")
                        del self.hooks[name]
                        self.hooks[name] = []# reset the hooks for the batch

            if batch_idx % self.config.batch_size == 0:                
                heatmaps = []
                for name in self.gradient:
                    heatmap = self.gradient[name]
                    # average the gradients over the batches but sum it over the channels                    
                    if heatmap.size != 0: #FIXE ME
                        plt.ioff()
                        heatmap = heatmap/(batch_idx+1)*self.config.batch_size 
                        heatmap = heatmap - np.min(heatmap)/np.max(heatmap)-np.min(heatmap)
                        ax = sns.heatmap(heatmap, cmap="rainbow",cbar=False)
                        plt.title(
                            f"Layer {trainer.model.erf_layers_names[name]}"
                        )
                        ax.set_axis_off()
                        heatmaps.append(wandb.Image(plt))
                        plt.close()

                if len(heatmaps) != 0 : 
                    wandb.log({f"heatmaps": heatmaps})
                
                
    def on_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:

        if (pl_module.current_epoch) % self.config.erf_freq == 0:
            
            self.gradient = {i: self.eps for i in range(self.layers)} #in case we wanna log on other epochs
            for hooks in pl_module.hooks:
                hooks.remove()
            pl_module.rq_grad = False


class LogAttentionVisualizationCallback(Callback):
    # TODO add attention vizualization for training and validation data
    # should probably use hooks to keep the model structure
    # classe image = classe predominante dans l'image + prendre features encoder
    # We shall use the activation map in order to visualize the effective receptive fields
    # those can be obtained by projecting the activation maps from upper layers
    # and resizing with a deconvolution or with interpolation ?
    # the original paper uses an average over 1000 images, this can also be considered.
    pass




class LogBarlowPredictionsCallback(Callback):

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Called when the training batch ends."""

        # `outputs` comes from `LightningModule.validation_step`
        # which corresponds to our model predictions in this case

        # Let's log 20 sample image predictions from first batch
        if batch_idx == 0:
            self.log_images("train", batch, 5, outputs)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Called when the training batch ends."""

        # `outputs` comes from `LightningModule.validation_step`
        # which corresponds to our model predictions in this case

        # Let's log 20 sample image predictions from first batch
        if batch_idx == 0:
            self.log_images("val", batch, 5, outputs)
            
    def log_images(self, name, batch, n, outputs):

        x1, x2 = batch
        image1 = x1[:n].cpu().detach().numpy()
        image2 = x2[:n].cpu().detach().numpy()

        samples1 = []
        samples2 = []
        mean = np.array([0.485, 0.456, 0.406])  # TODO this is not beautiful
        std = np.array([0.229, 0.224, 0.225])

        for i in range(n):

            bg1 = image1[i].transpose((1, 2, 0))
            bg1 = std * bg1 + mean
            bg1 = np.clip(bg1, 0, 1)


            bg2 = image2[i].transpose((1, 2, 0))
            bg2 = std * bg2 + mean
            bg2 = np.clip(bg2, 0, 1)

            samples1.append(wandb.Image(bg1))
            samples2.append(wandb.Image(bg2))
            
        wandb.log({f"{name}/x1": samples1})
        wandb.log({f"{name}/x2":samples2}) #TODO merge graphs   

    
