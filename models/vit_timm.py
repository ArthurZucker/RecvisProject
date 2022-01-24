import timm

def vit_timm(vit_parameters):
    name = vit_parameters["name"]
    pretrained = vit_parameters["pretrained"]
    model =  timm.create_model(name, pretrained)
    return model