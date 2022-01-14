import os
from turtle import forward

import torch
import torch.nn as nn
from torchvision.transforms.functional import InterpolationMode
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import torchvision.datasets 
from utils.transforms import toLongTensor, SegTransform
import datasets
from utils.agent_utils import get_net,get_head
from vit_pytorch.extractor import Extractor
class Segmentation(LightningDataModule):
    def __init__(self, config, dataset_name="VOCSegmentation"):
        super().__init__()
        
        self.network_param = config.network_param
        self.optim_param = config.optim_param
        self.lr = self.optim_param.lr
        
        
        if dataset_name == "VOCSegmentation":
            self.dataset = getattr(torchvision.datasets, dataset_name)
            if self.config.root_dataset is not None:
                self.root = self.config.root_dataset
            else:
                self.root = os.path.join(self.config.asset_path, "VOC")
        else: # use custom dataset : 
            raise NotImplementedError
            self.dataset = getattr(datasets, dataset_name)
            self.root = os.path.join(self.config.asset_path, dataset_name)
        
        self.batch_size = self.config.dataset_param.batch_size

        self.transform = self.get_transforms(config.dataset_param.input_size)

        # intialize the backbone 
        self.backbone = get_net(
            self.network_param.backbone, self.network_param.backbone_parameters
        )
        # load weights. here state dic keys should be taken care of
        if self.network_param.weight_checkpoint is not None: 
            pth = torch.load(self.network_param.weight_checkpoint)
            state_dict = { k.replace('backbone.','') : v for k,v in pth['state_dict'].items()}
            self.backbone.load_state_dict(state_dict, strict = False)
            print(f"Loaded checkpoints from {self.network_param.weight_checkpoint}")
            if self.network_param.backbone == "vit":
                self.backbone =  Extractor(self.backbone, return_embeddings_only=True)
                
        if self.network_param.backbone_parameters is not None:
            self.patch_size = self.network_param.backbone_parameters["patch_size"]
        self.in_features = list(self.backbone.modules())[-1].in_features

        # import mlp head
        self.head = get_head(self.network_param.head,self.network_param.head_params)
        

    def get_transforms(self,input_size):
        # @TODO should the mean be somewhere else?
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        return {
            "train": SegTransform(
                input_size,
                0.5,
                0.5,
                mean,
                std,
            ),
            "val": {
                "data": transforms.Compose(
                    [
                        transforms.Resize(input_size),
                        transforms.ToTensor(),
                        transforms.Normalize(mean, std),
                    ]
                ),
                "target": transforms.Compose(
                    [
                        transforms.Resize(
                            input_size, interpolation = InterpolationMode.NEAREST
                        ),
                        transforms.ToTensor(),
                        toLongTensor(),
                    ]
                ),
            },
        }
    # OPTIONAL, called for every GPU/machine (assigning state is OK)
    def setup(self, stage=None):
        # transforms
        # split dataset
        if stage in (None, "fit"):
            self.voc_train = self.dataset(
                self.root, image_set="train", transforms=self.transform["train"]
            )
            self.voc_val = self.dataset(
                self.root,
                image_set="val",
                transform=self.transform["val"]["data"],
                target_transform=self.transform["val"]["target"],
            )
            # Split between train and valid set (80/20)
            # val_length =int(len(init_dataset)*self.config.split_val)
            # lengths = [len(init_dataset)-val_length,val_length]
            # self.voc_train, self.voc_val = random_split(init_dataset, lengths)
        if stage == "test":
            self.voc_test = self.dataset(
                self.root,
                image_set="val",
                transform=self.transform["val"]["data"],
                target_transform=self.transform["val"]["target"],
            )
        if stage == "predict":
            # during prediction, the logginf is disabled
            self.voc_predict = self.dataset(
                self.root,
                image_set="val",
                transform=self.transform["val"]["data"],
                target_transform=self.transform["val"]["target"],
            )

    def forward(self,x):
        # @TODO @FIXME dimension will never be alright, has to correspond to the backbone> ViT should only use the extracor
        x = self.backbone(x)
        x = self.head(x)
        return x
        
        
    def train_dataloader(self):
        voc_train = DataLoader(
            self.voc_train,
            batch_size=self.batch_size,
            num_workers=self.config.num_workers,
            shuffle=True,
        )
        return voc_train

    def val_dataloader(self):
        voc_val = DataLoader(
            self.voc_val,
            batch_size=self.batch_size,
            num_workers=self.config.num_workers,
        )
        return voc_val

    def test_dataloader(self):
        voc_test = DataLoader(
            self.voc_test,
            batch_size=self.batch_size,
            num_workers=self.config.num_workers,
        )
        return voc_test

    def predict_dataloader(self):
        voc_predict = DataLoader(
            self.voc_predict,
            batch_size=self.batch_size,
            num_workers=self.config.num_workers,
        )
        return voc_predict
