import pytorch_lightning as pl
import wandb
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint, RichProgressBar)
from utils.callbacks import (LogAttentionMapsCallback,
                             LogBarlowCCMatrixCallback,
                             LogBarlowPredictionsCallback,
                             LogERFVisualizationCallback, LogMetricsCallback,
                             LogSegmentationCallback)

from agents.BaseTrainer import BaseTrainer


class trainer(BaseTrainer):
    def __init__(self, config, run):
        super().__init__(config, run)
        if "Seg" in config.hparams.datamodule:
            self.metric_param = config.metric_param
        self.callback_param = config.callback_param
        self.batch_size = config.data_param.batch_size

    def run(self):
        super().run()
        trainer = pl.Trainer(
            logger=self.wb_run,  # W&B integration
            callbacks=self.get_callbacks(),
            gpus=self.config.gpu,  # use all available GPU's
            max_epochs=self.config.max_epochs,  # number of epochs
            precision=self.config.precision,  # train in half precision
            accelerator="auto",
            check_val_every_n_epoch=self.config.val_freq,
            fast_dev_run=self.config.dev_run,
            accumulate_grad_batches=self.config.accumulate_size,
            log_every_n_steps=1,
            # default_root_dir=f"{wandb.run.name}",
        )
        trainer.logger = self.wb_run
        trainer.fit(self.model, datamodule=self.datamodule)

    def get_callbacks(self):

        callbacks = [
            RichProgressBar(),
            LearningRateMonitor(),
        ]

        if "Barlo" in self.config.arch:
            callbacks += [
                LogBarlowPredictionsCallback(self.callback_param.log_pred_freq), LogBarlowCCMatrixCallback(
                    self.callback_param.log_ccM_freq),
            ]

        if "vit" in self.encoder :
            callbacks += [
                LogAttentionMapsCallback(
                    self.callback_param.attention_threshold,
                    self.callback_param.nb_attention,
                    self.callback_param.log_att_freq
                )
            ]

        if "Seg" in self.config.datamodule:
            callbacks += [
                # LogERFVisualizationCallback(
                #     self.callback_param.nb_erf,
                #     self.callback_param.log_erf_freq,
                #     self.batch_size,
                # ),
                LogMetricsCallback(self.metric_param),
                LogSegmentationCallback(self.callback_param.log_pred_freq),
                EarlyStopping(monitor="val/loss", patience=10,mode="min", verbose=True),
            ]
            monitor = "val/iou"
            mode = "max"
        else:
            monitor = "val/loss"
            mode = "min"
        wandb.define_metric(monitor, summary=mode)
        if "Dino" in self.config.arch:
            save_top_k = -1
            every_n_epochs = 20
        else:
            save_top_k = 5
            every_n_epochs = 1

        if self.config.test:  # don't need to save if we are just testing
            save_top_k = 0

        callbacks += [
            ModelCheckpoint(
                monitor=monitor,
                mode=mode,
                verbose=True,
                dirpath= f"{self.config.weights_path}/{str(wandb.run.name)}",
                save_top_k=save_top_k,
                every_n_epochs=every_n_epochs,
            )
        ]  # our model checkpoint callback

        return callbacks
