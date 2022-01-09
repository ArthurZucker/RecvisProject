import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    RichProgressBar,
    LearningRateMonitor,
    EarlyStopping,
)
from utils.callbacks import (
    LogAttentionMapsCallback,
    LogBarlowCCMatrixCallback,
    LogBarlowImagesCallback,
    LogDinoImagesCallback,
    LogERFVisualizationCallback,
    LogMetricsCallback,
)
from agents.BaseTrainer import BaseTrainer
import wandb


class trainer(BaseTrainer):
    def __init__(self, config, CallBackParams, MetricsParams, run):
        super().__init__(config, run)
        self.MetricsParams = MetricsParams
        self.CallBackParams = CallBackParams

    def run(self):
        super().run()
        trainer = pl.Trainer(
            logger=self.wb_run,  # W&B integration
            callbacks=self.get_callbacks(self.CallBackParams, self.MetricsParams),
            gpus=self.config.gpu,  # use all available GPU's
            max_epochs=self.config.max_epochs,  # number of epochs
            precision=self.config.precision,  # train in half precision
            accelerator="auto",
            check_val_every_n_epoch=self.config.val_freq,
            fast_dev_run=self.config.dev_run,
            accumulate_grad_batches=self.config.accumulate_size,
            log_every_n_steps=1,
            default_root_dir=f"{wandb.run.name}",
        )
        trainer.logger = self.wb_run
        trainer.fit(self.model, datamodule=self.datamodule)

    def get_callbacks(self):

        callbacks = [RichProgressBar(), LearningRateMonitor()]

        if "Barlo" in self.config.arch:
            callbacks += [
                LogBarlowImagesCallback(self.config.log_pred_freq),
                LogBarlowCCMatrixCallback(self.config.log_ccM_freq),
            ]

        elif self.config.arch == "Dino" or self.config.arch == "DinoTwins":
            callbacks += [LogDinoImagesCallback(self.config.log_pred_freq)]

        if self.encoder == "vit":
            callbacks += [
                LogAttentionMapsCallback(
                    self.config.attention_threshold, self.config.nb_attention
                )
            ]

        if "Seg" in self.config.datamodule:
            callbacks += [
                LogMetricsCallback(),
                LogERFVisualizationCallback(self.config.log_erf_freq),
                EarlyStopping(monitor="val/loss", patience=4, mode="min", verbose=True),
            ]
            monitor = "val/iou"
            mode = "max"
        else:
            monitor = "val/loss"
            mode = "min"
        self.run.define_metric(monitor, summary=mode)
        if "Dino" in self.config.arch:
            save_top_k = -1
            every_n_epochs = 20
        else:
            save_top_k = 5
            every_n_epochs = 1

        if self.config.testing:  # don't need to save if we are just testing
            save_top_k = 0

        callbacks += [
            ModelCheckpoint(
                monitor=monitor,
                mode=mode,
                filename="{epoch:02d}-{val/loss:.2f}",
                verbose=True,
                dirpath=self.config.weights_path + f"/{str(wandb.run.name)}",
                save_top_k=save_top_k,
                every_n_epochs=every_n_epochs,
            )
        ]  # our model checkpoint callback

        return callbacks
