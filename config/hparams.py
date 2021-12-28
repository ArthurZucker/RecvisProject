import os
import os.path as osp
from dataclasses import dataclass
from posixpath import split
from typing import List, Dict, ClassVar, Optional, Tuple
import numpy as np
from simple_parsing.helpers import dict_field, list_field
import torch
import simple_parsing
from simple_parsing import choice
import random

"""Dataclass allows to have arguments and easily use wandb's sweep module.

Each arguments can be preceded by a comment, which will describe it when you call 'main.pu --help
An example of every datatype is provided. Some of the available arguments are also available.
Most notably, the agent, dataset, optimizer and loss can all be specified and automatically parsed
"""


@dataclass
class Hparams:
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
    #             # ignore_index=ignore_index,
    #             weight=torch.cat((torch.tensor([0.5]), torch.ones(21))),
    #         )
    #     )
    # )
    loss: Dict[str, Dict[str, str]] = dict_field(
        dict(
            models_losses_segmentation_models_dice_DiceLoss=dict(
                mode="multiclass",
                # classes=[i for i in range(0,21)],
            )
        )
    )  # FIXME add a loss dataclass
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
    lr: float = 0.08
    # agent to use for training
    agent: str = "BT_trainer"
    # architecture to use
    arch: str = "BarlowTwins"
    # encoder for barlow
    encoder: str = "resnet50"
    # data module
    datamodule: str = "BarlowTwinsVOCSegmentationDataModule"
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
    accumulate_size: int = 512
    # save directory
    save_dir: str = osp.join(os.getcwd(), "wandb")
    # number of workers for dataloaders
    num_workers: int = 16
    # tune the model on first run
    tune_lr: bool = False
    # tune the model on first run
    tune_batch_size: bool = False
    # number or gpu
    gpu: int = 1
    # precision
    precision: int = 32
    # effective receptive fields log frequency
    erf_freq: int = 20
    # index of the layers to use for the receptive field visualization
    layers: int = 5 #  TODO number of layers to plot a repartition ex quartiles 25%, 50% etx
    # metrics
    metrics: Dict[
        str, Dict[str, str]
    ] = dict_field(  # TODO use simple parsing inheritance, list of various dataclasses
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
            ConfusionMatrix=dict(
                num_classes=n_classes,
                normalize='true'
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

    # optimizer
    optimizer: Dict[str, Dict[str, str]] = dict_field(
        dict(torch_optim_SGD=dict(momentum=0.9, nesterov=False))
    )

    # scheduler
    scheduler: Dict[str, Dict[str, str]] = dict_field(
        dict(
            torch_optim_lr_scheduler_ReduceLROnPlateau=dict(
                patience=4, mode="max", threshold=0.1
            )
        )
    )
    # set to 0 to remove validation
    limit_val_batches: int = 1.0
    # numbeer of projection channels 
    bt_proj_channels: int = 2048
    # lambda barlow twins
    lmbda: int = 1
    # log images during val and tarin frequency (in epochs) : 
    log_pred_freq : int = 10

@dataclass
class DatasetParams:
    """Dataset Parameters"""

    default_root: ClassVar[str] = "/dataset"  # the default root directory to use.

    dataset: str = "CIFAR10"  # laptop,pistol
    """ dataset name: only [cifar10] for now """

    root_dir: str = default_root  # dataset root directory


@dataclass
class NetworkParams:
    # Network parameters
    encoder_type: str = choice(
        "resnet50", "swinT", "swinS", "resnet", default="resnet50"
    )  # One of: mlp, cnn, dcgan, resnet # try resnet :)

# TODO config for each task (choice barlow or segmentation) ti be able to run them


@dataclass
class OptimizerParams:
    """Optimization parameters"""

    optimizer: str = "adam"  # Optimizer (adam, rmsprop)
    lr: float = 0.0001  # learning rate, default=0.0002
    lr_sched_type: str = "step"  # Learning rate scheduler type.
    z_lr_sched_step: int = 100000  # Learning rate schedule for z.
    lr_iter: int = 10000  # Learning rate operation iterations
    normal_lr_sched_step: int = 100000  # Learning rate schedule for normal.
    z_lr_sched_gamma: float = 1.0  # Learning rate gamma for z.
    normal_lr_sched_gamma: float = 1.0  # Learning rate gamma for normal.
    normal_consistency_loss_weight: float = 1e-3  # Normal consistency loss weight.
    z_norm_weight_init: float = 1e-2  # Normal consistency loss weight.
    z_norm_activate_iter: float = 1000  # Normal consistency loss weight.
    spatial_var_loss_weight: float = 1e-2  # Spatial variance loss weight.
    grad_img_depth_loss: float = 2.0  # Spatial variance loss weight.
    spatial_loss_weight: float = 0.5  # Spatial smoothness loss weight.
    beta1: float = 0.0  # beta1 for adam. default=0.5
    n_iter: int = 76201  # number of iterations to train
    batchSize: int = 4  # input batch size
    alt_opt_zn_interval: Optional[int] = None
    """ Alternating optimization interval.
    - None: joint optimization
    - 20: every 20 iterations, etc.
    """
    alt_opt_zn_start: int = 100000
    """Alternating optimization start interation.
    - -1: starts immediately,
    - '100: starts alternating after the first 100 iterations.
    """


@dataclass
class Parameters:
    """base options."""

    # Dataset parameters.
    dataset: DatasetParams = DatasetParams()
    # Set of parameters related to the optimizer.
    optimizer: OptimizerParams = OptimizerParams()
    # GAN Settings
    hparams: Hparams = Hparams()

    def __post_init__(self):
        """Post-initialization code"""
        # Mostly used to set some values based on the chosen hyper parameters
        # since we will use different models, backbones and datamodules

        # Set render number of channels
        if self.hparams.arch == "BarlowTwins":
            self.hparams.limit_val_batches = 0  # TODO later we might need to do something
            self.hparams.lr = 0.04
            self.hparams.bt_proj_channels = 2048
            
        # Set random seed
        if self.hparams.seed_everything is None:
            self.hparams.seed_everything = random.randint(1, 10000)
        print("Random Seed: ", self.hparams.seed_everything)
        random.seed(self.hparams.seed_everything)
        torch.manual_seed(self.hparams.seed_everything)
        if not self.hparams.gpu != 0:
            torch.cuda.manual_seed_all(self.hparams.seed_everything)

    @classmethod
    def parse(cls):
        parser = simple_parsing.ArgumentParser()
        parser.add_arguments(cls, dest="parameters")
        args = parser.parse_args()
        instance: Parameters = args.parameters
        return instance
