import torch.nn as nn
import torch
from math import sqrt
from einops.layers.torch import Rearrange
class Baseline(nn.Module):
    """
    Simple baseline head for semantic segmentation
    """

    def __init__(self, config):
        super().__init__()
        decoder_hidden_size = config.decoder_hidden_size
        input_dim = config.input_dim
        img_size = config.img_size[0]
        patch_size = config.patch_size
        self.num_labels = config.num_labels
        self.img_size   = img_size
        self.patch_size = patch_size
        # just a linear layer of the flattened output of the embeddings
        # designed to use ViT outputs but should also work with resnet50
        # if input is [B,C,Patch], flatten to [B,CxPatch]
        
        self.to_reconstructed = nn.Sequential(
            nn.Linear(decoder_hidden_size, (patch_size **2) * self.num_labels),
            Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h = (img_size // patch_size), w =  (img_size // patch_size), p1 = patch_size, p2 = patch_size, c=self.num_labels),
        )
        
    def forward(self, hidden_states: torch.Tensor):
        
        # hidden_states = hidden_states.permute(0, 2, 1) # [B,emb_dim,nb_patch + 1]
        # hidden_states = hidden_states.permute(0,2,1)
        # hidden_states = self.proj(torch.flatten(hidden_states,start_dim=1))
        # h = hidden_states.reshape(batch_size, -1, (self.img_size[0]//16), (self.img_size[0]//16))
        # h = nn.functional.interpolate(     # used with ksize of one
        #     h, size=self.img_size, mode="bilinear", align_corners=False
        # )
        
        # @TODO check that the reshape is correctly, use cls token attention maps as another aditionnal feature that can be extracted from it
        hidden_states = hidden_states[:,1:,:]
        h = self.to_reconstructed (hidden_states)
        return h
        # encoder_hidden_state = hidden_states.permute(0, 2, 1)
        # hidden_states = hidden_states[:,:,1:].reshape(batch_size, -1, (self.img_size[0]//self.patch_size), (self.img_size[0]//self.patch_size))
        # # upsample
        
        # logits = self.classifier(hidden_states)
        # logits = nn.functional.interpolate(
        #     logits, size=self.img_size, mode="bilinear", align_corners=False
        # )
        return logits
