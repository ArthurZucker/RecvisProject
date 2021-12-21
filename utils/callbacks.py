import wandb
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger


class LogPredictionsCallback(Callback):
    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Called when the validation batch ends."""

        # `outputs` comes from `LightningModule.validation_step`
        # which corresponds to our model predictions in this case

        # Let's log 20 sample image predictions from first batch
        if batch_idx == 0:
            n = 20
            x, y = batch
            images = [img for img in x[:n]]
            captions = [
                f"Ground Truth: {y_i} - Prediction: {y_pred}"
                for y_i, y_pred in zip(y[:n], outputs[:n])
            ]
            # Option 1: log images with `WandbLogger.log_image`
            trainer.logger.log_image(key="sample_images", images=images, caption=captions)
            # Option 2: log predictions as a Table
            columns = ["image", "ground truth", "prediction"]
            data = [
                [wandb.Image(x_i), y_i, y_pred]
                for x_i, y_i, y_pred in list(zip(x[:n], y[:n], outputs[:n]))
            ]
            trainer.logger.log_table(key="sample_table", columns=columns, data=data)

        # TODO add sample input image visualization 

class LogFeatureVisualizationCallback(Callback):      
    pass 
    # TODO add feature vizualization for training and validation data
    # should probably use hooks to keep the model structure 