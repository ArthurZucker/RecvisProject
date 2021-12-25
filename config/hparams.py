import os
import os.path as osp
from dataclasses import dataclass
from posixpath import split
from typing import List, Dict
import numpy as np
from simple_parsing.helpers import dict_field, list_field
import torch

"""Dataclass allows to have arguments and easily use wandb's weep module.

Each arguments can be preceded by a comment, which will describe it when you call 'main.pu --help
An example of every datatype is provided. Some of the available arguments are also available.
Most notably, the agent, dataset, optimizer and loss can all be specified and automatically parsed
"""
#TODO log learning rate

@dataclass
class hparams:
    """Hyperparameters of the Model"""

    # weight and biases
    wandb_entity: str = "recvis"
    # projectname
    wandb_project: str = "test-sem-seg"
    # seed
    seed_everything: float = 40  # np.random.randint(10000)
    # maximum number of epochs
    max_epochs: int = 40
    # path to download pascal voc
    asset_path: str = osp.join(os.getcwd(), "assets")
    # ignore class
    ignore_index: int = 21
    # loss to train the model
    # loss: Dict[str, Dict[str, str]] = dict_field(
    #     dict(
    #         torch_nn_CrossEntropyLoss=dict(
    #             ignore_index=ignore_index,
    #             weight=torch.cat((torch.tensor([0.5]), torch.ones(21))),
    #         )
    #     )
    # )
    loss: Dict[str, Dict[str, str]] = dict_field(
        dict(
            models_losses_segmentation_models_jaccard_JaccardLoss=dict(
                mode="multiclass",
                # classes=[i for i in range(0,21)],
            )
        )
    )  # FIXME le modele n'apprends pas bcp quelque soit la loss
    # loss: Dict[str, Dict[str, str]] = dict_field(
    #     dict(
    #         models_losses_segmentation_models_focal_FocalLoss=dict(
    #             mode="multiclass",
    #             ignore_index=ignore_index,
    #         )
    #     )
    # )
    # resize coefficients for H and w
    input_size: tuple = (256, 256)
    # learning rate
    lr: float = 0.0001
    # agent to use for training
    agent: str = "BaseTrainer"
    # architecture to use
    arch: str = "deeplabv3"
    # data module
    datamodule: str = "VOCSegmentationDataModule"
    # classes
    n_classes: int = 21
    # number of channels
    n_channels: int = 3
    # batch size for training
    batch_size: int = 16
    # split value
    split_val: float = 0.2
    # validation frequency
    val_freq: int = 1
    # developpment mode, only run 1 batch of train val and test
    dev_run: bool = False
    # gradient accumulation batch size
    accumulate_size: int = 16
    # save directory
    save_dir: str = osp.join(os.getcwd(), "wandb")
    # number of workers for dataloaders
    num_workers: int = 16
    # tune the model on first run
    tune: bool = False
    # number or gpu
    gpu: int = 1
    # precision
    precision: int = 32
    # effective receptive fields log frequency
    erf_freq: int = 20
    # index of the layers to use for the receptive field visualization
    layers: List[int] = list_field(
        64, 128, 150
    )  # 182 is lqst TODO takle a repartition ex quartiles 25%, 50% etx
    # metrics
    metrics: Dict[str, Dict[str, str]] = dict_field(
        dict(
            Accuracy=dict(
                num_classes=n_classes,
                average="weighted",
                mdmc_average="global",
                # ignore_index=ignore_index,
            ),
            Recall=dict(
                num_classes=n_classes,
                average="weighted",
                mdmc_average="global",
                # ignore_index=ignore_index,
            ),
            Precision=dict(
                num_classes=n_classes,
                average="weighted",
                mdmc_average="global",
                # ignore_index=ignore_index,
            ),
            F1=dict(
                num_classes=n_classes,
                average="weighted",
                mdmc_average="global",
                # ignore_index=ignore_index,
            ),
            #     AveragePrecision=dict(
            #         num_classes=n_classes, average="weighted", ignore_index=ignore_index
            # ),
            IoU=dict(
                num_classes=n_classes,
                # ignore_index=ignore_index
            ),
        )
    )
    # ConfusionMatrix=dict()))

    # optimizer
    optimizer: Dict[str, Dict[str, str]] = dict_field(dict(torch_optim_AdamW=dict()))

    # scheduler
    scheduler: Dict[str, Dict[str, str]] = dict_field(
        dict(
            torch_optim_lr_scheduler_ExponentialLR=dict(
                gamma=0.5
            )
        )
    )
