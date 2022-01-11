"""
Auteur: Clement Apavou  & Arthur Zucker
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
    wdb_config = {}
    for k,v in vars(parameters).items():
        for key,value in vars(v).items():
            wdb_config[f"{k}-{key}"]=value
    wandb_run = WandbLogger(
        config=wdb_config,# vars(parameters),  # FIXME use the full parameters
        project=parameters.hparams.wandb_project,
        entity=parameters.hparams.wandb_entity,
        allow_val_change=True,
        save_dir=parameters.hparams.save_dir,
    )
    config = parameters
    # Create the Agent and pass all the configuration to it then run it..
    agent_class = globals()[config.hparams.agent]
    agent = agent_class(config, wandb_run)
    agent.run()


if __name__ == "__main__":
    main()
