from torchvision.models.segmentation import deeplabv3_resnet50
import torch.nn as nn
import torch
from vit_pytorch import ViT
from vit_pytorch.extractor import Extractor
import models.decode_heads as decoder
from einops.layers.torch import Rearrange
import timm

vit_dino_dict = {"vitsdino8": "vit_small_patch8_224_dino", "vitsdino16": "vit_small_patch16_224_dino",
                 "vitbdino8": "vit_base_patch8_224_dino", "vitbdino16": "vit_base_patch16_224_dino"}


class SemanticModel(nn.Module):
    def __init__(self, config) -> None:
        super(SemanticModel, self).__init__()
        """
        backbone_pretrained : ImageNet, VOC, 
        name_encoder : resnet50, vit, vitsdino8, vitsdino16, vitbdino8, vitbdino16
        name_head : Linear, SETRnaive, SETRPUP, DeepLabHead
        """
        self.config = config
        num_classes = self.config.encoder_param['n_classes']
        self.name_encoder = config.backbone
        self.name_head = config.head if hasattr(config, 'head') else None 

        if self.name_encoder == "resnet50":
            if self.name_head is None :
                # deeplabv3 deeplab head pretrain + resnet50 imagenet classification
                self.net = deeplabv3_resnet50(
                    pretrained=False, num_classes=num_classes, pretrained_backbone=True)

                temp_net = deeplabv3_resnet50(
                    pretrained=True, num_classes=num_classes, pretrained_backbone=True)

                self.net.classifier = temp_net.classifier
            else:
                self.net = deeplabv3_resnet50(
                    pretrained=self.config.model_param['pretrained'], num_classes=num_classes, pretrained_backbone=self.config.model_param['pretrained_backbone'])

                if hasattr(self.config, "weight_checkpoint_backbone"):
                    pth = torch.load(
                        self.config.weight_checkpoint_backbone, map_location=torch.device('cpu'))
                    if "resnet50.pth" not in self.config.weight_checkpoint_backbone:
                        pth = {k.replace('backbone.', ''): v for k,
                               v in pth['state_dict'].items()}
                    self.net.backbone.load_state_dict(pth, strict=False)

            # Freeze backbone weights
            if self.config.encoder_param['freeze']:
                for param in self.net.backbone.parameters():
                    param.requires_grad = False

        elif self.name_encoder == "vit":  # @TODO get_net using backbone_parameters
            self.vit = ViT(**self.config.backbone_parameters)

            if hasattr(self.config, "weight_checkpoint_backbone"):
                pth = torch.load(
                    self.config.weight_checkpoint_backbone, map_location=torch.device('cpu'))
                state_dict = {
                    k.replace('backbone.', ''): v for k, v in pth['state_dict'].items()}
                self.vit.load_state_dict(state_dict, strict=False)

            # for name, param in self.vit.named_parameters(): print(f"{name} : {param}")
            self.vit = Extractor(self.vit, return_embeddings_only=True)

            if self.config.model_param['pretrained']:
                net = deeplabv3_resnet50(
                    pretrained=self.config.model_param['pretrained'], num_classes=num_classes, pretrained_backbone=self.config.model_param['pretrained_backbone'])
                self.classifier = net.classifier

            # Freeze backbone weights
            if self.config.model_param['freeze']:
                for param in self.vit.parameters():
                    param.requires_grad = False

        elif self.name_encoder in vit_dino_dict:  # @TODO get_net using backbone_parameters

            self.vit = timm.create_model(
                vit_dino_dict[self.name_encoder], pretrained=self.config.head_param['pretrained'])

            self.vit.head = nn.Identity()

            if hasattr(self.config, "weight_checkpoint_backbone"):
                pth = torch.load(
                    self.config.weight_checkpoint_backbone, map_location=torch.device('cpu'))
                state_dict = {
                    k.replace('backbone.', ''): v for k, v in pth['state_dict'].items()}
                self.vit.load_state_dict(state_dict, strict=False)

            if "vits" in self.name_encoder:
                embedding_dim = 384
            else:
                embedding_dim = 768
            if "8" in self.name_encoder:
                self.patch_size = 8
            else:
                self.patch_size = 16

            input_size = 224

            if self.name_head == "Linear":
                self.head = nn.Sequential(
                    nn.Linear(embedding_dim, self.patch_size *
                              self.patch_size*num_classes),
                    Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=(input_size // self.patch_size),
                              w=(input_size // self.patch_size), p1=self.patch_size, p2=self.patch_size, c=num_classes),
                )

            elif self.name_head == "DeepLabHead":
                self.head = decoder.MyDeepLabHead(
                    embedding_dim=embedding_dim, patch_dim=self.patch_size, img_dim=input_size, num_classes=num_classes)

            elif self.name_head == "SETRPUP":
                self.head = decoder.SETR_PUP(
                    embedding_dim=embedding_dim, patch_dim=self.patch_size, img_dim=input_size, num_classes=num_classes)

            elif self.name_head == "SETRnaive":
                self.head = decoder.SETR_Naive(
                    embedding_dim=embedding_dim, patch_dim=self.patch_size, img_dim=input_size, num_classes=num_classes)

            else:
                raise ValueError(f'Head {self.name_head} not supported')

            # self.head = decoder.SETR_MLA(
            #     embedding_dim=384, patch_dim=8, img_dim=224, num_classes=num_classes) # doses'nt work
            # self.head = decoder.MLAHead(mla_channels=256, mlahead_channels=128, norm_cfg=dict(
            #     type='SyncBN', requires_grad=True)) # doses'nt work

            # Freeze backbone weights
            if self.config.encoder_param['freeze']:
                for param in self.vit.parameters():
                    param.requires_grad = False
        else:
            raise ValueError(
                f'option encoder {self.name_encoder} does not exist !')

    def forward(self, x):
        
        if self.name_encoder == "resnet50" or self.name_encoder is None:
            dic = self.net(x)
            return dic['out']

        elif "vit" in self.name_encoder:
            embeddings = self.vit(x)
            x = self.head(embeddings)
            return x
