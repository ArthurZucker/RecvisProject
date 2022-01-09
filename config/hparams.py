import os
import random
from dataclasses import dataclass
from os import path as osp
from typing import Any, ClassVar, Dict, List, Optional

import numpy as np
import simple_parsing
import torch
import torch.optim
from simple_parsing.helpers import Serializable, choice, dict_field, list_field


@dataclass
class Hparams:
    """Hyperparameters of for the run"""

    
    wandb_project : str  = "recvis"         # name of the project
    test          : bool = True             # test code before running
    wandb_entity  : str  = (f"{'test-'*test}sem-seg")       # name of the wandb entity, here our team
    save_dir      : str  = osp.join(os.getcwd(), "wandb")   # directory to save wandb outputs


    agent       : str           = "BT_trainer"      # trainer agent to use for training
    arch        : str           = "BarlowTwins"     # architecture to use
    datamodule  : str           = "DinoDataModule"  # lighting datamodule
    dataset     : Optional[str] = "CIFAR10"         # dataset, use <Dataset>Eval for FT
    weights_path: str           = osp.join(os.getcwd(), "weights") # path to save weights
    asset_path  : str           = osp.join(os.getcwd(), "assets")  # path to download datasets
        
    seed_everything: Optional[int] = None   # seed for the whole run
    tune_lr        : bool          = False  # tune the model on first run
    tune_batch_size: bool          = False  # tune the model on first run
    gpu            : int           = 1      # number or gpu
    precision      : int           = 32     # precision
    val_freq       : int           = 1      # validation frequency
    accumulate_size: int           = 1024   # gradient accumulation batch size
    max_epochs     : int           = 400    # maximum number of epochs
    dev_run        : bool          = False  # developpment mode, only run 1 batch of train val and test


@dataclass
class DatasetParams:
    """Dataset Parameters
    ! The batch_size and number of crops should be defined here
    """
    
    num_workers       : int         = 20         # number of workers for dataloadersint
    input_size        : tuple       = (32, 32)   # image_size
    batch_size        : int         = 128        # batch_size
    asset_path        : str         = osp.join(os.getcwd(), "assets")  # path to download the dataset
    n_crops           : int         = 5          # number of crops/global_crops
    n_global_crops    : int         = 2          # number of global crops
    global_crops_scale: List[int]   = list_field(0.5, 1)      # scale range of the global crops
    local_crops_scale : List[float] = list_field(0.05, 0.5)   # scale range of the local crops
    # @TODO the numbner of classes should be contained in the dataset and extracted automatically for the network?

@dataclass
class OptimizerParams:
    """Optimization parameters"""

    optimizer           : str            = "AdamW"  # Optimizer (adam, rmsprop)
    lr                  : float          = 5e-4     # learning rate,                             default = 0.0002
    lr_sched_type       : str            = "step"   # Learning rate scheduler type.
    min_lr              : float          = 5e-6     # minimum lr for the scheduler
    betas               : List[float]    = list_field(0.9, 0.999)  # beta1 for adam. default = (0.9, 0.999)
    warmup_epochs       : int            = 10
    scheduler_parameters: Dict[str, Any] = dict_field(
        dict(
            base_value         = 0.9995,
            final_value        = 1,
            max_epochs         = 0,
            niter_per_ep       = 0,
            warmup_epochs      = 0,
            start_warmup_value = 0,
        )
    )
    lr_scheduler_parameters: Dict[str, Any] = dict_field(
        dict(
            base_value         = 0,
            final_value        = 0,
            max_epochs         = 0,
            niter_per_ep       = 0,
            warmup_epochs      = 10,
            start_warmup_value = 0,
        )
    )

@dataclass
class CallBackParams:
    """Parameters to use for the logging callbacks
    """
    log_erf_freq       : int   = 10     # effective receptive fields
    log_att_freq       : int   = 10     # attention maps
    log_pred_freq      : int   = 10     # log_pred_freq
    log_ccM_freq       : int   = 1      # log cc_M matrix frequency
    log_dino_freq      : int   = 1      # log output frrequency for dino
    attention_threshold: float = 0.5    # Logging attention threshold for head fusion
    nb_attention       : int   = 5      # nb of images for which the attention will be visualised

@dataclass
class MetricsParams:
    num_classes : int       = 21        # number of classes for the segmentation task
    average     : str       = "weighted"
    mdmc_average: str       = "global"
    ignore_index: int       = 21
    names       : List[str] = list_field("Accuracy","Recall","Precision","F1","IoU") # name of the metrics which will be used

    

@dataclass
class BarlowConfig:
    """Hyperparameters specific to Barlow Twin Model.
    Used when the `arch` option is set to "Barlow" in the hparams
    """
    
    # lambda coefficient used to scale the scale of the redundancy loss
    # so it doesn't overwhelm the invariance loss
    lmbda                 : float        = 5e-3
    bt_proj_channels      : int          = 2048      # number of channels to use for projection
    encoder               : str          = choice("resnet50", "swinS", default="resnet50") # encoder for barlow
    pretrained_encoder    : bool         = False     # use a pretrained model
    use_backbone_features : bool         = True      # only use backbone features for FT
    weight_checkpoint     : Optional[str]= osp.join(os.getcwd(),) # model checkpoint used in classification fine tuning
    
    


@dataclass
class DinoConfig:
    """Hyperparameters specific to the DINO Model.
    Used when the `arch` option is set to "Barlow" in the hparams
    """
    backbone                  : str               = "vit" 
    proj_layers               : int               = 3
    proj_channels             : int               = 2048
    bottleneck_dim            : int               = 256
    out_channels              : int               = 4096
    warmup_teacher_temp_epochs: int               = 10  # Default 30
    center_momentum           : float             = 0.9  # Default 0.9
    student_temp              : float             = 0.1
    teacher_temp              : float             = 0.07  # Default 0.04, can be linearly increased to 0.07 but then it becomes unstable
    warmup_teacher_temp       : float             = (0.04  )# would be different from techer temp if we used a warmup for this param
    backbone_parameters       : Optional[str]     = None
    if backbone == "vit":
        backbone_parameters: Dict[str, Any]    = dict_field(
                dict(
                    image_size  = 32,
                    patch_size  = 4,
                    num_classes = 0,
                    dim         = 192,
                    depth       = 4,
                    heads       = 6,
                    mlp_dim     = 1024,
                    dropout     = 0.1,
                    emb_dropout = 0.1,
                )
        )
    weight_checkpoint  : Optional[str] = osp.join(os.getcwd(),)
    backbone_parameters: Optional[str] = None

    if backbone == "vit":
        backbone_parameters: Dict[str, Any]    = dict_field(
                dict(
                    image_size  = 32,
                    patch_size  = 4,
                    num_classes = 0,
                    dim         = 192,
                    depth       = 4,
                    heads       = 6,
                    mlp_dim     = 1024,
                    dropout     = 0.1,
                    emb_dropout = 0.1,
                )
        )



@dataclass
class Parameters:
    """base options."""
    hparams    : Hparams         = Hparams()
    optim_param: OptimizerParams = OptimizerParams()

    def __post_init__(self):
        """Post-initialization code"""
        # Mostly used to set some values based on the chosen hyper parameters
        # since we will use different models, backbones and datamodules

        # Set render number of channels
        if "BarlowTwins" in self.hparams.arch:
            self.network_param: BarlowConfig = BarlowConfig()
        elif "Dino" in self.hparams.arch:
            self.network_param: DinoConfig = DinoConfig()
        # Set random seed
        if self.hparams.seed_everything is None:
            self.hparams.seed_everything = random.randint(1, 10000)

        self.data_param: DatasetParams = DatasetParams()
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
