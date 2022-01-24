import pytorch_lightning as pl
from utils.agent_utils import get_datamodule, get_lightning_module


class BaseTrainer:
    def __init__(self, config, run) -> None:
        self.config = config.hparams
        self.wb_run = run
        self.model = get_lightning_module(config.hparams.arch, config)
        self.encoder = config.network_param.backbone

        self.wb_run.watch(self.model, log_graph=False)
        self.datamodule = get_datamodule(
            config.hparams.datamodule, config
        )

    def run(self):

        if self.config.tune_batch_size:
            trainer = pl.Trainer(
                logger=self.wb_run,
                gpus=self.config.gpu,
                auto_scale_batch_size="power",
                accelerator="auto",
                default_root_dir=self.wb_run.save_dir
            )
            trainer.logger = self.wb_run
            trainer.tune(self.model, datamodule=self.datamodule)

        if self.config.tune_lr:
            trainer = pl.Trainer(
                logger=self.wb_run,
                gpus=self.config.gpu,
                auto_lr_find=True,
                accelerator="auto",
                default_root_dir=self.wb_run.save_dir
            )
            trainer.logger = self.wb_run
            trainer.tune(self.model, datamodule=self.datamodule)
