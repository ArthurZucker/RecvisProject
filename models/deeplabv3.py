# deeplabv3_resnet50 does not work that well
from torchvision.models.segmentation import deeplabv3_resnet101, deeplabv3_resnet50
from torchvision.models.segmentation.deeplabv3 import DeepLabV3, DeepLabHead
import torch.nn as nn
from torchvision.models import resnet50
import torch
from vit_pytorch import ViT
from vit_pytorch.extractor import Extractor
"""
https://pytorch.org/vision/stable/models.html#torchvision.models.segmentation.deeplabv3_resnet50
"""


class Deeplabv3(nn.Module):
    def __init__(self, num_classes, pretrained=False, backbone=None, freeze=False) -> None:
        super(Deeplabv3, self).__init__()
        self.name_encoder = backbone

        if self.name_encoder is None:
            # model pre-trained on COCO train2017 which contains the same classes as Pascal VOC
            self.net = deeplabv3_resnet101(
                pretrained=pretrained, num_classes=num_classes)
            self.name_encoder = None
        else:

            # option 2 :
            if self.name_encoder == "resnet50":
                self.net = deeplabv3_resnet50(
                    pretrained=pretrained, num_classes=num_classes)
                pth = torch.load(
                    "/home/clement/Documents/Cours/MVA/S1/Cours_to_validate/RECVIS_2021/Projet/RecvisProject/weights/barlow_twins/resnet50.pth", map_location=torch.device('cpu'))
                self.net.backbone.load_state_dict(pth, strict=False)

                # just add deeplabhead
                # self.net = resnet50(pretrained=pretrained)
                # pth = torch.load("/home/clement/Documents/Cours/MVA/S1/Cours_to_validate/RECVIS_2021/Projet/RecvisProject/weights/barlow_twins/resnet50.pth", map_location=torch.device('cpu'))
                # self.net.load_state_dict(pth, strict=False)
                # features = self.net.fc.in_features
                # # 1
                # self.net.fc = nn.Identity()
                # self.net.fc = DeepLabHead(features, num_classes)
                # 2 
                # self.net = DeepLabV3(backbone=self.encoder, classifier=DeepLabHead(features, num_classes))
                # self.net.backbone = encoder
                # self.net.classifier = DeepLabHead(features, num_classes)

                # Freeze backbone weights
                if freeze:
                    for param in self.net.backbone.parameters():
                        param.requires_grad = False
            elif self.name_encoder == "vit":
                self.vit = ViT(
                        image_size=(128, 128),
                        patch_size=128//8,
                        dim=768,
                        heads=6,
                        depth=4,
                        mlp_dim=1,
                        num_classes=21,  
                        )
                self.vit.mlp_head = nn.Identity()
                self.vit = Extractor(self.vit)
                self.classifier = DeepLabHead(768, num_classes)

                # Freeze backbone weights
                if freeze:
                    for param in self.vit.parameters():
                        param.requires_grad = False
            else:
                raise ValueError(f'option encoder {self.name_encoder} does not exist !')

    def forward(self, x):
        if self.name_encoder != "vit":
            dic = self.net(x)
            return dic['out']
        else:
            _, embeddings = self.vit(x)
            x = self.classifier(embeddings.unsqueeze(3).permute(0,2,1,3))
            return x