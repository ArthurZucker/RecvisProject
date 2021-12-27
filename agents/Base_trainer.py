import pytorch_lightning as pl
from utils.agent_utils import get_datamodule, get_net
from utils.logger import init_logger


class BaseTrainer:
    def __init__(self, config, run) -> None:
        self.config = config
        self.wb_run = run
        self.model = get_net(config)
        self.wb_run.watch(self.model)
        self.datamodule = get_datamodule(config)
        self.logger = init_logger("Trainer", "DEBUG")

    def run(self):
        
        if self.config.tune_lr:
            trainer = pl.Trainer(
                logger=self.wb_run,
                gpus=self.config.gpu,
                auto_lr_find=True,
                accelerator="auto",
            )
            trainer.logger = self.wb_run
            trainer.tune(self.model, datamodule=self.datamodule)
        
        if self.config.tune_batch_size:
            trainer = pl.Trainer(
                logger=self.wb_run,
                gpus=self.config.gpu,
                auto_scale_batch_size="power",
                accelerator="auto",
            )
            trainer.logger = self.wb_run
            trainer.tune(self.model, datamodule=self.datamodule)
