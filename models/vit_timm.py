import timm

def vit_timm(vit_parameters, pretrained=False):
    name = vit_parameters["name"]
    model =  timm.create_model(name, pretrained=pretrained)
    return model