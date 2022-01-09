from pytorch_pretrained_vit import ViT

def Vit( vit_parameters,freeze = False) -> None:
    net = ViT(**vit_parameters) # @TODO filllatere
    if freeze:
        for param in net.parameters():
            param.requires_grad = False
    return net
