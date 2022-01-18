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
    def __init__(self, config) -> None:
        super(Deeplabv3, self).__init__()
        self.config = config
        num_classes = self.config.model_param['n_classes']
        self.name_encoder = config.backbone

        if self.name_encoder is None:
            # model pre-trained on COCO train2017 which contains the same classes as Pascal VOC
            self.net = deeplabv3_resnet101(
                pretrained=self.config.model_param.pretrained, num_classes=num_classes)
            self.name_encoder = None

        else:

            if self.name_encoder == "resnet50":
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
                if self.config.model_param['freeze']:
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
                else:
                    self.classifier = DeepLabHead(
                        self.config.backbone_parameters['dim'], num_classes)

                # Freeze backbone weights
                if self.config.model_param['freeze']:
                    for param in self.vit.parameters():
                        param.requires_grad = False
            else:
                raise ValueError(
                    f'option encoder {self.name_encoder} does not exist !')

    def forward(self, x):
        if self.name_encoder != "vit":
            dic = self.net(x)
            return dic['out']
        else:
            embeddings = self.vit(x)
            embeddings = embeddings[:, 1:, :].reshape(
                embeddings.shape[0], embeddings.shape[2], (embeddings.shape[1]-1)//8, (embeddings.shape[1]-1)//8)

            x = self.classifier(embeddings)
            x = nn.functional.upsample(x, scale_factor=32, mode='bilinear')
            return x
