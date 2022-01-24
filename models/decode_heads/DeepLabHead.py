import torch.nn as nn
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from einops.layers.torch import Rearrange

class MyDeepLabHead(nn.Module):

    def __init__(self, embedding_dim, patch_dim, img_dim, num_classes):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.patch_dim = patch_dim
        self.img_dim = img_dim
        self.num_classes = num_classes
        self.deeplabhead = DeepLabHead(self.embedding_dim, num_classes)

        self.head = nn.Sequential(
            nn.Linear(embedding_dim, self.patch_size *
                        self.patch_size*num_classes),
            Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=(self.img_dim // self.patch_size),
                     w=(self.img_dim // self.patch_size), p1=self.patch_size, p2=self.patch_size, c=num_classes),
        )
        
        self.upsample = nn.Upsample(
            scale_factor=self.patch_dim, mode='bilinear'
        )

    def forward(self, x):
        x = self._reshape_output(x)
        x = self.deeplabhead(x)
        x = self.upsample(x)
        return x

    def _reshape_output(self, x):
        x = x.view(
            x.size(0),
            int(self.img_dim / self.patch_dim),
            int(self.img_dim / self.patch_dim),
            self.embedding_dim,
        )
        x = x.permute(0, 3, 1, 2).contiguous()
        return x