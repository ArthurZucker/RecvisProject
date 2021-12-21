import pytorch_lightning as pl
from utils.logger import init_logger
from pytorch_lightning.callbacks import ModelCheckpoint
from utils.callbacks import LogPredictionsCallback


class Base_Trainer:
    def __init__(self, config, run) -> None:
        super().__init__()
        self.config = config
        self.wb_run = run
        self.model = get_net(config)
        print(self.model)
        self.wb_run.watch(self.model)

        self.logger = init_logger("Trainer", "DEBUG")

        trainer = pl.Trainer(auto_lr_find=True)
        trainer.tune(self.model, datamodule=self.config.datamodule)
        checkpoint_callback = ModelCheckpoint(monitor="val_accuracy", mode="max")

        # ------------------------
        # 3 INIT TRAINER
        # ------------------------
        # trainer = pl.Trainer.from_argparse_args(self.config)

        trainer = pl.Trainer(
            logger=self.run,  # W&B integration
            callbacks=[
                checkpoint_callback,  # our model checkpoint callback
                LogPredictionsCallback(),
            ],  # logging of sample predictions
            gpus=-1,  # use all available GPU's
            max_epochs=self.config.max_epochs,  # number of epochs
            precision=16,  # train in half precision
            deterministic=True,
        )

        trainer.fit(self.model, datamodule=self.config.datamodule)

