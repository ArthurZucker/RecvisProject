import torch.nn as nn
import torch
from math import sqrt

class Baseline(nn.Module):
    """
    Simple baseline head for semantic segmentation
    """

    def __init__(self, config):
        super().__init__()
        decoder_hidden_size = config.decoder_hidden_size
        input_dim = config.input_dim
        img_size = config.img_size
        patch_size = config.patch_size
        num_labels = config.num_labels
        self.img_size   = img_size
        self.patch_size = patch_size
        # just a linear layer of the flattened output of the embeddings
        # designed to use ViT outputs but should also work with resnet50
        # if input is [B,C,Patch], flatten to [B,CxPatch]
        
        # self.proj = nn.Linear(input_dim*((img_size[0]//patch_size)**2+1), decoder_hidden_size)
        self.proj = nn.ConvTranspose2d(decoder_hidden_size, num_labels, kernel_size=32,stride=30)
        
        # self.classifier = nn.Conv2d(decoder_hidden_size, num_labels, kernel_size=1)
        
    def forward(self, hidden_states: torch.Tensor):
        batch_size, n ,_  = hidden_states.shape
        # hidden_states = hidden_states.permute(0, 2, 1) # [B,emb_dim,nb_patch + 1]
        #hidden_states = self.proj(torch.flatten(hidden_states,start_dim=1))
        hidden_states = hidden_states[:,1:,:].permute(0,2,1)
        h = self.proj(hidden_states.reshape(batch_size, -1, (self.img_size[0]//self.patch_size), (self.img_size[0]//self.patch_size)))
        # h = nn.functional.interpolate(
        #     h, size=self.img_size, mode="bilinear", align_corners=False
        # )
        return h
        # encoder_hidden_state = hidden_states.permute(0, 2, 1)
        # hidden_states = hidden_states[:,:,1:].reshape(batch_size, -1, (self.img_size[0]//self.patch_size), (self.img_size[0]//self.patch_size))
        # # upsample
        
        # logits = self.classifier(hidden_states)
        # logits = nn.functional.interpolate(
        #     logits, size=self.img_size, mode="bilinear", align_corners=False
        # )
        return logits
