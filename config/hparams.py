import os
import os.path as osp
from dataclasses import dataclass
from posixpath import split
from typing import List, Dict
import numpy as np
from simple_parsing.helpers import dict_field, list_field

"""Dataclass allows to have arguments and easily use wandb's weep module.

Each arguments can be preceded by a comment, which will describe it when you call 'main.pu --help
An example of every datatype is provided. Some of the available arguments are also available.
Most notably, the agent, dataset, optimizer and loss can all be specified and automatically parsed
"""


@dataclass
class hparams:
    """Hyperparameters of the Model"""

    # weight and biases
    wandb_entity: str = "recvis"
    # projectname
    wandb_project: str = "test-sem-seg"
    # seed
    seed_everything: float = np.random.randint(10000)
    # maximum number of epochs
    max_epochs: int = 40
    # path to download pascal voc
    asset_path: str = osp.join(os.getcwd(), "assets")
    # # loss to train the model
    # loss: str = "CrossEntropy"
    # learning rate
    lr: float = 0.02089296130854041
    # agent to use for training
    agent: str = "Base_Trainer"
    # architecture to use
    arch: str = "unet"
    # data module
    datamodule: str = "VOCSegmentationDataModule"
    # classes
    n_classes: int = 22
    # number of channels
    n_channels: int = 3
    # use bilinear interpolation
    bilinear: bool = True
    # batch size for training
    batch_size: int = 16
    # split value
    split_val: float = 0.2
    # validation frequency
    val_freq: int = 1
    # developpment mode, only run 1 batch of train val and test
    dev_run: bool = False
    # gradient accumulation batch size
    accumulate_size: int = 32
    # save directory
    save_dir: str = osp.join(os.getcwd(), "wandb")
    # number of workers for dataloaders
    num_workers: int = 16
    # tune the model on first run
    tune: bool = False
    # number or gpu
    gpu: int = 1
    # precision
    precision: int = 16
    # effective receptive fields log frequency
    erf_freq: int = 2
    # index of the layers to use for the receptive field visualization
    layers: List[int] = list_field(64,80,95)
    # metrics
    metrics: Dict[str, Dict[str, str]] = dict_field(dict(Accuracy=dict(
        num_classes=n_classes, average="weighted", mdmc_average='global'
    ),
        Recall=dict(
            num_classes=n_classes, average="weighted", mdmc_average='global'
    ),
        Precision=dict(
            num_classes=n_classes, average="weighted", mdmc_average='global'
    ),
    #     AveragePrecision=dict(
    #         num_classes=n_classes, average="weighted", 
    # ),
        IoU=dict(
            num_classes=n_classes,
    )))
        # ConfusionMatrix=dict()))

    # optimizer
    optimizer: Dict[str, Dict[str, str]] = dict_field(dict(torch_optim_SGD=
                                                        dict()))
