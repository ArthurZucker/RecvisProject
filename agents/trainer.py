import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from utils.agent_utils import get_datamodule, get_net
from utils.callbacks import LogERFVisualizationCallback, LogPredictionsCallback, LogMetricsCallback
from utils.logger import init_logger


class Base_Trainer:
    def __init__(self, config, run) -> None:
        super().__init__()
        self.config = config
        self.wb_run = run
        self.model = get_net(config)
        print(self.model)
        self.wb_run.watch(self.model)
        self.datamodule = get_datamodule(config)
        self.logger = init_logger("Trainer", "DEBUG")

    def run(self):
        if self.config.tune:
            trainer = pl.Trainer(
                logger=self.wb_run,
                gpus=self.config.gpu,
                auto_scale_batch_size="power",
                accelerator="auto",
            )
            trainer.logger = self.wb_run
            trainer.tune(self.model, datamodule=self.datamodule)
            trainer = pl.Trainer(
                logger=self.wb_run,
                gpus=self.config.gpu,
                auto_lr_find=True,
                accelerator="auto",
            )
            trainer.logger = self.wb_run
            trainer.tune(self.model, datamodule=self.datamodule)
        
        checkpoint_callback = ModelCheckpoint(monitor="val/iou", mode="max")

        # TODO feature hook for feature fizualization, for every
        # should be implemented as a callback?
        # self.activation = np.array([])
        # self.feature_hook = self.model.net.fc.register_forward_hook(self.getActivation(f'{self.model.net.fc}'))

        # ------------------------
        # 3 INIT TRAINER
        # ------------------------
        # trainer = pl.Trainer.from_argparse_args(self.config)

        trainer = pl.Trainer(
            logger=self.wb_run,  # W&B integration
            callbacks=[
                checkpoint_callback,  # our model checkpoint callback
                LogPredictionsCallback(),
                LogERFVisualizationCallback(self.config),
                RichProgressBar(),
                LogMetricsCallback(self.config),
                EarlyStopping(monitor="val/iou"),
            ],  # logging of sample predictions
            gpus=self.config.gpu,  # use all available GPU's
            max_epochs=self.config.max_epochs,  # number of epochs
            precision=self.config.precision,  # train in half precision
            accelerator="auto",
            check_val_every_n_epoch=self.config.val_freq,
            fast_dev_run=self.config.dev_run,
            accumulate_grad_batches=self.config.accumulate_size,
            log_every_n_steps=1,
            # detect_anomaly = True,
        )
        trainer.logger = self.wb_run
        trainer.fit(self.model, datamodule=self.datamodule)
