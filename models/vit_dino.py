import timm
from timm.models.vision_transformer import default_cfgs
def vit_dino(vit_parameters):
    
    model =  timm.create_model('deit_small_patch16_224', pretrained=False)
    model.backbone_parameters = default_cfgs['vit_small_patch8_224_dino']
    return model