import wandb
from pytorch_lightning.callbacks import Callback
import numpy as np
import torch

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

        for i in range(len(batch)):

            bg_image = images[i].numpy().transpose((1, 2, 0))
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
        self.metrics_module_train = MetricsModule("train", config.metrics, config.n_classes)
        self.metrics_module_validation = MetricsModule("val", config.metrics)

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Called when the train batch ends."""

        x, y = batch
        self.metrics_module_train.update_metrics(outputs["logits"], y)

    def on_train_epoch_end(self, trainer, pl_module):
        """Called when the train epoch ends."""

        self.metrics_module_train.log_metrics("train/", trainer.logger)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Called when the validation batch ends."""

        x, y = batch
        self.metrics_module_validation.update_metrics(outputs["logits"], y)

    def on_validation_epoch_end(self, trainer, pl_module):
        """Called when the validation epoch ends."""

        self.metrics_module_validation.log_metrics("val/", trainer.logger)


class LogFeatureVisualizationCallback(Callback):
    # TODO add feature vizualization for training and validation data
    # should probably use hooks to keep the model structure
    # classe image = classe predominante dans l'image + prendre features encoder
    pass


class LogAttentionVisualizationCallback(Callback):
    # TODO add attention vizualization for training and validation data
    # should probably use hooks to keep the model structure
    # classe image = classe predominante dans l'image + prendre features encoder
    # We shall use the activation map in order to visualize the effective receptive fields
    # those can be obtained by projecting the activation maps from upper layers
    # and resizing with a deconvolution or with interpolation ?
    # the original paper uses an average over 1000 images, this can also be considered.
    pass
