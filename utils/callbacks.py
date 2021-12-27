
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

        for i in range(n):

            bg_image = images[i].detach().numpy().transpose((1, 2, 0))
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
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
        self.gradient = {i: 0 for i in self.layers}

    # from different stages of the network)
    # Our implementation should be network independant
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:
        if (pl_module.current_epoch%self.config.erf_freq == 0 and batch_idx == 0):
            pl_module._register_layer_hooks()

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        # hooks have already been placed to extract the output
        # here, compute the gradient of the various ouputs with respect
        # to the input feature map, average it over the validation batches
        # until we have the number of images we required
        if not trainer.sanity_checking and pl_module.current_epoch%self.config.erf_freq == 0:
            self.hooks = pl_module.features
            x, _ = batch
            for name in self.layers:
                # for each layer, compute the mean
                gradient_wrt_ipt = self.input_grad(name)
                if gradient_wrt_ipt != []:
                    try:
                        gradient_wrt_ipt = grad(gradient_wrt_ipt, x, retain_graph=True)[
                            0
                        ].detach()

                        self.gradient[name] += (np.squeeze(np.abs(torch.mean(torch.sum(gradient_wrt_ipt, axis= 1),axis=0).cpu()).numpy())-self.gradient[name])/(batch_idx+1)
                            # average over the batches but sum over the channels 

                        
                        del self.hooks[name]
                        self.hooks[name] = []# reset the hooks for the batch
                    except Exception as e:
                        # the gradient can't be computed because of batchnorm, jsut ignore
                        print(f"Tried to compute gradient error : {e}")
                        del gradient_wrt_ipt
                        del self.hooks[name]
                        self.hooks[name] = []# reset the hooks for the batch

            if batch_idx == [trainer.limit_train_batches-1, self.config.batch_size][trainer.limit_train_batches>0]:
                heatmaps = []
                for name in self.gradient:
                    plt.ioff()
                    # average the gradients over the batches but sum it over the channels
                    heatmap = self.gradient[name]
                    if heatmap != []:
                        heatmap = heatmap - np.min(heatmap)/np.max(heatmap)-np.min(heatmap)
                        ax = sns.heatmap(heatmap, cmap="viridis",cbar=False)
                        plt.title(
                            f"Layer {[ k for k,v in trainer.model.named_modules()][name]}"
                        )
                        ax.set_axis_off()
                        heatmaps.append(wandb.Image(plt))
                        plt.close()
                if heatmaps != [] : 
                    wandb.log({f"heatmaps": heatmaps})
                self.gradient = {i: [] for i in self.layers}
                
                if pl_module.current_epoch %self.config.erf_freq == 0:
                    for hooks in pl_module.hooks:
                        hooks.remove()
                    self.gradient = {i: 0 for i in self.layers}
                    pl_module.rq_grad = False

    def input_grad(self, name):
        return self.hooks[name]  # returns the output of the feature map

    

class LogAttentionVisualizationCallback(Callback):
    # TODO add attention vizualization for training and validation data
    # should probably use hooks to keep the model structure
    # classe image = classe predominante dans l'image + prendre features encoder
    # We shall use the activation map in order to visualize the effective receptive fields
    # those can be obtained by projecting the activation maps from upper layers
    # and resizing with a deconvolution or with interpolation ?
    # the original paper uses an average over 1000 images, this can also be considered.
    pass
