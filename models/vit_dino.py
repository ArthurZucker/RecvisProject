import timm
from timm.models.vision_transformer import default_cfgs

def vit_timm(vit_parameters, pretrained=False):
    name = vit_parameters["name"]
    model =  timm.create_model(name, pretrained=pretrained)
    return model