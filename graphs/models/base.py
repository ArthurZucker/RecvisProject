from graphs.weights_initializer import weights_init
from graphs.models.custom_layers.layer_norm import LayerNorm

import torch.nn as nn

class Base(nn.Module):
    
    def __init__(self,options):
        super(Base,self).__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError
