import timm
from timm.models.vision_transformer import default_cfgs
def vit_dino(vit_parameters):
    name = vit_parameters["name"]
    model =  timm.create_model(name, pretrained=False)
    return model