import pytorch_lightning as pl
from utils.logger import init_logger
from pytorch_lightning.callbacks import ModelCheckpoint
from utils.callbacks import LogPredictionsCallback
from utils.agent_utils import get_net, get_datamodule
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


class Base_Trainer:
    def __init__(self, config, run) -> None:
        super().__init__()
        self.config = config
        self.wb_run = run
        self.model = get_net(config)
        print(self.model)
        self.wb_run.watch(self.model)

        self.logger = init_logger("Trainer", "DEBUG")

        trainer = pl.Trainer(
            logger=self.wb_run, gpus=1, auto_lr_find=True, accelerator="auto"
        )
        self.datamodule = get_datamodule(config)
        trainer.tune(self.model, datamodule=self.datamodule)
        checkpoint_callback = ModelCheckpoint(monitor="val_accuracy", mode="max")

        # ------------------------
        # 3 INIT TRAINER
        # ------------------------
        # trainer = pl.Trainer.from_argparse_args(self.config)

        trainer = pl.Trainer(
            logger=self.wb_run,  # W&B integration
            callbacks=[
                checkpoint_callback,  # our model checkpoint callback
                LogPredictionsCallback(),
                EarlyStopping(monitor="val_loss"),
            ],  # logging of sample predictions
            gpus=1,  # use all available GPU's
            max_epochs=self.config.max_epochs,  # number of epochs
            precision=16,  # train in half precision
            deterministic=True,
            accelerator="auto",
            check_val_every_n_epoch=self.config.val_freq,
            fast_dev_run=self.config.dev_run,
        )

        trainer.fit(self.model, datamodule=self.datamodule)
