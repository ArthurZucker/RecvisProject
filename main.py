"""
Auteur: Arthur Zucker
- Capture configuration
- Update with argvs
- launches training through agent
"""
from __future__ import absolute_import, division

from pytorch_lightning.loggers import WandbLogger

from agents import *
from config.hparams import Parameters


def main():
    # from apex import amp
    parameters = Parameters.parse()
    # initialize wandb instance
    wandb_run = WandbLogger(
        config=vars(parameters.hparams),  # FIXME use the full parameters
        project=parameters.hparams.wandb_project,
        entity=parameters.hparams.wandb_entity,
        allow_val_change=True,
        save_dir=parameters.hparams.save_dir,
    )
    config = wandb_run.experiment.config
    # seed_everything(config.seed_everything)

    # Create the Agent and pass all the configuration to it then run it..
    agent_class = globals()[config.agent]
    agent = agent_class(config, wandb_run)
    # run the model
    agent.run()
    # agent.finalize()


if __name__ == "__main__":
    main()
