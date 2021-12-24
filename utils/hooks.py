import torch
def get_activation(name, features):
    def hook(model, input, output):
        if output.requires_grad:
            mid_x= output.shape[-1]
            mid_y= output.shape[-2] 
            # average over the batches but sum over the channels    
            features[name].append(torch.mean(torch.sum(output[:, :, mid_x//2, mid_y//2],dim = 1),axis=0))
        # TODO only take the center pixel of the output feature map to compute
        # FIXME replace = with append (in order to then compute the average over all layers?)

    return hook
