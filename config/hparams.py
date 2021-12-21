from dataclasses import dataclass
from posixpath import split
from typing import List
import numpy as np
from simple_parsing.helpers import list_field

import os, os.path as osp
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
    seed_everything: float = 40
    # maximum number of epochs 
    max_epochs: int = 40
    # path to download pascal voc
    asset_path: str = osp.join(os.getcwd(), "assets")
    # loss to train the model 
    loss: str = "CrossEntropy"
    # learning rate 
    lr : float = 0.03
    # agent to use for training
    agent: str = "Base_Trainer"
    # architecture to use
    arch : str = "base_voc"
    # data module 
    datamodule: str = "VOCSegmentationDataModule"
    # classes 
    n_classes : int = 21
    # number of channels
    n_channels: int = 3
    # use bilinear interpolation 
    bilinear: bool = True
    # batch size for training
    batch_size: int = 8
    # split value 
    split_val: float = 0.2
    # validation frequency 
    val_freq: int = 1