import timm
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from torchvision.models.segmentation import deeplabv3_resnet50
from vit_pytorch import ViT
from vit_pytorch.extractor import Extractor

import models.heads as heads

vit_dino_dict = {"vitsdino8": "vit_small_patch8_224_dino", "vitsdino16": "vit_small_patch16_224_dino",
                 "vitbdino8": "vit_base_patch8_224_dino", "vitbdino16": "vit_base_patch16_224_dino"}


class SemanticModel(nn.Module):
    def __init__(self, config) -> None:
        super(SemanticModel, self).__init__()
        """
        backbone_pretrained : ImageNet, VOC, 
        name_backbone : resnet50, vit, vitsdino8, vitsdino16, vitbdino8, vitbdino16
        name_head : Linear, SETRnaive, SETRPUP
        """
        self.config = config
        num_classes = self.config.head_params['n_classes']

        self.name_backbone = config.backbone
        self.name_head = config.head

        if self.name_backbone == "resnet50":
            if self.name_head is None:
                # deeplabv3 deeplab head pretrain + resnet50 imagenet classification
                self.net = deeplabv3_resnet50(
                    pretrained=False, num_classes=num_classes, pretrained_backbone=True)

                temp_net = deeplabv3_resnet50(
                    pretrained=True, num_classes=num_classes, pretrained_backbone=True)

                self.net.classifier = temp_net.classifier
            else:
                self.net = deeplabv3_resnet50(
                    pretrained=self.config.head_params['pretrained'], pretrained_backbone=self.config.backbone_params['pretrained'], num_classes=num_classes)

                if self.config.checkpoint_backbone is not None:
                    pth = torch.load(
                        self.config.checkpoint_backbone, map_location=torch.device('cpu'))
                    if "resnet50.pth" not in self.config.checkpoint_backbone:
                        pth = {k.replace('backbone.', ''): v for k,
                               v in pth['state_dict'].items()}
                    self.net.backbone.load_state_dict(pth, strict=False)
                    print(
                        f"Loaded checkpoints from {self.config.checkpoint_backbone}")

        elif self.name_backbone == "vit_pytorch":
            self.vit = ViT(**self.config.backbone_parameters)

            if self.config.checkpoint_backbone is not None:
                pth = torch.load(
                    self.config.checkpoint_backbone, map_location=torch.device('cpu'))
                state_dict = {
                    k.replace('backbone.', ''): v for k, v in pth['state_dict'].items()}
                self.vit.load_state_dict(state_dict, strict=False)
                print(
                    f"Loaded checkpoints from {self.config.checkpoint_backbone}")

            self.vit = Extractor(self.vit, return_embeddings_only=True)

            input_size = self.config.backbone_parameters['image_size']
            embedding_dim = self.config.backbone_parameters['dim']
            self.patch_size = self.config.backbone_parameters['patch_size']

        elif self.name_backbone in vit_dino_dict:

            self.vit = timm.create_model(
                vit_dino_dict[self.name_backbone], pretrained=self.config.backbone_params['pretrained'])

            self.vit.head = nn.Identity()

            if self.config.checkpoint_backbone is not None:
                pth = torch.load(
                    self.config.checkpoint_backbone, map_location=torch.device('cpu'))
                state_dict = {
                    k.replace('backbone.', ''): v for k, v in pth['state_dict'].items()}
                self.vit.load_state_dict(state_dict, strict=False)
                print(
                    f"Loaded checkpoints from {self.config.checkpoint_backbone}")

            input_size = 224
            if "vits" in self.name_backbone:
                embedding_dim = 384
            else:
                embedding_dim = 768
            if "8" in self.name_backbone:
                self.patch_size = 8
            else:
                self.patch_size = 16

        else:
            raise ValueError(
                f'Backbone {self.name_backbone} not supported !')

        # Freeze backbone weights
        if self.config.backbone_params['freeze']:
            if "vit" in self.name_backbone:
                for param in self.vit.parameters():
                    param.requires_grad = False
            else:
                for param in self.net.backbone.parameters():
                    param.requires_grad = False

        # Choose decoder head for ViT
        if "vit" in self.name_backbone:
            if self.name_head == "Linear":
                self.head = nn.Sequential(
                    nn.Linear(embedding_dim, self.patch_size *
                              self.patch_size*num_classes),
                    Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=(input_size // self.patch_size),
                              w=(input_size // self.patch_size), p1=self.patch_size, p2=self.patch_size, c=num_classes),
                )

            elif self.name_head == "SETRPUP":
                self.head = heads.SETR_PUP(
                    embedding_dim=embedding_dim, patch_dim=self.patch_size, img_dim=input_size, num_classes=num_classes)

            elif self.name_head == "SETRnaive":
                self.head = heads.SETR_Naive(
                    embedding_dim=embedding_dim, patch_dim=self.patch_size, img_dim=input_size, num_classes=num_classes)

            else:
                raise ValueError(f'Head {self.name_head} not supported !')

    def forward(self, x):

        if self.name_backbone == "resnet50":
            dic = self.net(x)
            return dic['out']

        elif "vit" in self.name_backbone:
            embeddings = self.vit(x)
            if self.name_backbone == "vit_pytorch":
                embeddings = embeddings[:, 1:]  # remove cls token
            x = self.head(embeddings)
            return x
