from dataclasses import dataclass
from typing import List
import numpy as np
from simple_parsing.helpers import list_field

import os
"""Dataclass allows to have arguments and easily use wandb's weep module.

Each arguments can be preceded by a comment, which will describe it when you call 'main.pu --help
An example of every datatype is provided. Some of the available arguments are also available.
Most notably, the agent, dataset, optimizer and loss can all be specified and automatically parsed
"""


@dataclass
class hparams:
    """Hyperparameters of Yout Model"""
    # validation frequency 
    validate_every: int = 1
    # num_classes
    num_classes: int = 20
    # feature_extracting or fine_tuning
    feature_extracting: bool = False
    # Learning rate of the Adam optimizer.
    lr: float = 1e-3
    # batch sier
    batch_size : int = 4
    # Use cuda for training
    cuda: bool = True
    # Architecture to choose, available are "denet (to come)", "sincnet (to come)", "leaf (to come)", "yolor (to come)"
    arch: str = "Contrastive_vit"
    # Agent to use, the agent has his own trining loop
    agent: str = "ContrastiveAgent"
    # Dataset used for training
    dataloader: str = "BirdsDataloader"
    # output file for kaggle
    outfile: str = "result.csv"
    # path to images in dataset
    image_dir: str = os.path.join(os.getcwd(),"assets/bird_dataset")
    # test directory
    test_dir: str = os.path.join(os.getcwd(),"assets/bird_dataset/test_images/mistery_category")
    # Number of workers used for the dataloader
    num_workers: int  = 16
    # weight_decay
    weight_decay: float = 0.0001
    # momentum 
    momentum: float = 0.8
    # seed
    seed: float = np.random.random()
    # gpu_device
    gpu_device : int = 0
    # optimizer
    optimizer: str = "SGD"

    # loss
    loss: str = "CrossEntropy"
    # checkpoint dir
    checkpoint_dir: str = os.path.join(os.getcwd(),"weights/")
    # checkpoint file
    checkpoint_file: str = "lilac-haze-214_model_best_89.81.pth.tar"#"solar-terrain-177_model_best_87.00.pth.tar"#"faithful-wood-176_model_best_86.00.pth.tar" #pious-dew-166_model_best_97.00.pth.tar" #"devout-planet-34_model_best_95.19.pth.tar"
    # mode
    mode: str = "train"
    # Toggle testing mode, which only runs a few epochs and val
    test_mode: bool = False
    # max epoch tu run
    max_epoch: int = 150
    # async_loading
    async_loading: bool = True
    # activation function
    activation: str = "relu"
    # accuracy threshold used for siames network 
    accuracy_threshold: float = 0.5
    # weighted samplet on dataloaders
    weighted_sampler: bool = True
    # example of list parameters
    # layer resolutions
    fc_lay: List[int] = list_field(2048, 2048, 2048, 1024, 256, 2)
    fc_drop: List[float] = list_field(0.0, 0.0, 0.0, 0.0, 0.0, 0.1)
    fc_use_laynorm: List[bool] = list_field(
        False, False, False, False, False, False)
    fc_act: List[str] = list_field(
        "leaky_relu", "linear", "leaky_relu", "leaky_relu", "leaky_relu", "softmax")
