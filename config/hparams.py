import os
import random
from dataclasses import dataclass
from os import path as osp
from typing import Any, ClassVar, Dict, List, Optional

import simple_parsing
import torch
import torch.optim
from simple_parsing.helpers import Serializable, choice, dict_field, list_field

################################## Global parameters ##################################

@dataclass
class Hparams:
    """Hyperparameters of for the run"""

    
    wandb_entity  : str  = "recvis"         # name of the project
    test          : bool = True            # test code before running, if testing, no checkpoints are written
    wandb_project : str  = (f"{'test-'*test}sem-seg")
    save_dir      : str  = osp.join(os.getcwd())   # directory to save wandb outputs


    agent       : str           = "trainer"             # trainer agent to use for training
    arch        : str           = "BarlowTwins"        # architecture to use
    datamodule  : str           = "BarlowTwins"        # lighting datamodule @TODO will soon be deleted since it is the same, get datamodule will use arch
    dataset     : Optional[str] = "BarlowTwinsDataset"     # dataset, use <Dataset>Eval for FT
    weights_path: str           = osp.join(os.getcwd(), "weights") # path to save weights
    asset_path  : str           = osp.join(os.getcwd(), "assets")  # path to download datasets
        
    seed_everything: Optional[int] = None   # seed for the whole run
    tune_lr        : bool          = False  # tune the model on first run
    tune_batch_size: bool          = False  # tune the model on first run
    gpu            : int           = 0      # number or gpu
    precision      : int           = 32     # precision
    val_freq       : int           = 1      # validation frequency
    accumulate_batch: int           = 8    # gradient accumulation batch size
    max_epochs     : int           = 800    # maximum number of epochs

    dev_run        : bool          = False  # developpment mode, only run 1 batch of train val and test


@dataclass
class DatasetParams:
    """Dataset Parameters
    ! The batch_size and number of crops should be defined here
    """
    
    num_workers       : int         = 20         # number of workers for dataloadersint
    input_size        : tuple       = (224, 224)   # image_size
    batch_size        : int         = 6        # batch_size

    asset_path        : str         = osp.join(os.getcwd(), "assets")  # path to download the dataset
    root_dataset      : Optional[str] = None
    # @TODO the numbner of classes should be contained in the dataset and extracted automatically for the network?


@dataclass
class CallBackParams:
    """Parameters to use for the logging callbacks
    """
    log_erf_freq       : int   = 1      # effective receptive fields
    nb_erf             : int   = 6
    log_att_freq       : int   = 1      # attention maps
    log_pred_freq      : int   = 10     # log_pred_freq
    log_pred_nb         : int  = 5
    log_ccM_freq       : int   = 1     # log cc_M matrix frequency
    attention_threshold: float = 0.6    # Logging attention threshold for head fusion
    nb_attention       : int   = 6      # nb of images for which the attention will be visualised
    early_stopping_params     : Dict[str, Any] = dict_field(
        dict(
            monitor="val/loss", 
            patience=50,
            mode="min",
            verbose=True
        )
    )
################################## Self-supervised learning parameters ##################################

class BarlowConfig:
    """Hyperparameters specific to Barlow Twin Model.
    Used when the `arch` option is set to "Barlow" in the hparams
    """
    
    # lambda coefficient used to scale the scale of the redundancy loss
    # so it doesn't overwhelm the invariance loss
    backbone              : str           = "vit_timm" # vit_pytorch or vit_timm
    nb_proj_layers        : int           = 3         # nb projection layers, defaults is 3 should not move
    lmbda                 : float         = 5e-2
    bt_proj_dim           : int           = 512      # number of channels to use for projection
    weight_checkpoint     : Optional[str] = None
    # weight_checkpoint     : Optional[str] = osp.join("weights", "leafy-water_epoch=394-step=9084.ckpt")
    # weight_checkpoint     : Optional[str] = osp.join("/kaggle/input/", "weights-barlow-twins/leafy-water_epoch394-step9084.ckpt")
    backbone_parameters   : Optional[str] = None

@dataclass
class OptimizerParams_SSL: # @TODO change name 
    """Optimization parameters"""

    optimizer           : str            = "AdamW"  # Optimizer (adam, rmsprop)
    lr                  : float          = 3e-4     # learning rate,                             default = 0.0002
    lr_sched_type       : str            = "step"   # Learning rate scheduler type.
    min_lr              : float          = 2.5e-4     # minimum lr for the scheduler 5e-6 for VIT works great
    betas               : List[float]    = list_field(0.9, 0.999)  # beta1 for adam. default = (0.9, 0.999)
    warmup_epochs       : int            = 5
    max_epochs          : int            = 1000      # @TODO duplicate of dataparam
    use_scheduler       : bool           = True
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

################################## Fine-tuning on segmentation task parameters ##################################

@dataclass
class SegmentationConfig:
    """Hyperparameters specific to the Segmentation Model.
    Used when the `arch` option is set to "Segmentation" in the hparams
    """
    # Deep Lab v3 model with resnet 50 (deeplabv3 or None)
    # backbone            : str          = "resnet50"
    # backbone_params      : Dict[str, Any] = dict_field(
    #     dict(
    #         freeze=True,
    #         pretrained=True,
    #     )
    # )
    # head                : str          = "deeplab"
    # head_params         : Dict[str, Any] = dict_field(
    #     dict(
    #         n_classes=21,
    #         pretrained=True
    #     )
    # )
    # ViT + heads (SETR PUP, SETR naive or Linear)
    # backbones available : vit_pytorch, vit, vitsdino8, vitsdino16, vitbdino8, vitbdino16
    backbone            : str          = "vit_pytorch"
    backbone_params      : Dict[str, Any] = dict_field(
        dict(
            freeze=True,
            pretrained=True,
        )
    )
    # heads available : Linear, SETRnaive, SETRPUP
    head                : str          = "SETRPUP"
    head_params         : Dict[str, Any] = dict_field(
        dict(
            n_classes=21,
        )
    )
    checkpoint_backbone : Optional[str] = None
    # checkpoint_backbone : Optional[str] = osp.join("weights", "rare_valley_epoch=330-step=7612.ckpt")


@dataclass
class LossParams: # @TODO remove classs, put in segmentaiton config 
    """Loss parameters"""
    name: str = "models.losses.segmentation.dice.DiceLoss"
    param: Dict[str, Any] = dict_field(dict())


@dataclass
class OptimizerParams_Segmentation:
    """Optimization parameters"""

    optimizer           : str            = "AdamW" 
    lr                  : float          = 5e-3
    max_epochs          : int            = 400      
    use_scheduler       : bool           = True


@dataclass
class MetricsParams:
    metrics     : List[str] = list_field("Accuracy", "Recall", "Precision", "F1", "IoU") # name of the metrics which will be used
    pixel_wise_parameters : Dict[str, Any] = dict_field(
        dict(
            average           = "weighted",
            mdmc_average      = "global"
        )
    )
    num_classes : int          = 21        # number of classes for the segmentation task


################################## Call correct parameters ##################################

@dataclass
class Parameters:
    """base options."""
    hparams       : Hparams         = Hparams()
    data_param    : DatasetParams   = DatasetParams()
    callback_param: CallBackParams  = CallBackParams()
    metric_param  : MetricsParams   = MetricsParams()
    loss_param    : LossParams      = LossParams()

    def __post_init__(self):
        """Post-initialization code"""
        # Mostly used to set some values based on the chosen hyper parameters
        # since we will use different models, backbones and datamodules
        self.hparams.wandb_project = (f"{'test-'*self.hparams.test}sem-seg") 
        
        if "BarlowTwins" in self.hparams.arch:
            self.network_param : BarlowConfig        = BarlowConfig()
            self.optim_param   : OptimizerParams_SSL = OptimizerParams_SSL()
        elif "Segmentation" in self.hparams.arch:
                self.network_param : SegmentationConfig = SegmentationConfig()
                self.optim_param   : OptimizerParams_Segmentation = OptimizerParams_Segmentation()
        else:
            raise ValueError(f'Architecture {self.hparams.arch} not supported !')
    
        # Set random seed
        if self.hparams.seed_everything is None:
            self.hparams.seed_everything = random.randint(1, 10000)
            
            
        if self.network_param.backbone=="vit_pytorch" :
            self.network_param.backbone_parameters = dict(
                image_size      = self.data_param.input_size[0],
                patch_size      = self.data_param.input_size[0]//25,
                num_classes     = 0,
                dim             = 384,
                depth           = 12,
                heads           = 6,
                mlp_dim         = 1024,
                dropout         = 0.1,
                emb_dropout     = 0.1,
            )
        
        if self.network_param.backbone=="vit_timm" :
            self.network_param.backbone_parameters = dict(
                image_size      = self.data_param.input_size[0],
                patch_size      = 16,
                dim             = 1000, # 384 for vit16 small, 1000 for deit small
                name = "deit_tiny_patch16_224",
                pretrained = True
            )
        
        
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
