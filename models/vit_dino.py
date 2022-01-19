import timm
from timm.models.vision_transformer import default_cfgs
def vit_dino(vit_parameters):
    
    model =  timm.create_model('vit_small_patch8_224_dino', pretrained=True)
    model.backbone_parameters = default_cfgs['vit_small_patch8_224_dino']
    return model