import torch

def get_activation(name, features):
    def hook(model, input, output):
        if output.requires_grad:
            # average over the batches but sum over the channels    
            if len(output.shape) == 3 : 
                output = output[:,1:,:].reshape(output.shape[0],output.shape[2],(output.shape[1]-1)//8,(output.shape[1]-1)//8)
            
            mid_x = output.shape[-1]
            mid_y = output.shape[-2] 
            features[name].append(torch.mean(torch.sum(output[:, :, mid_x//2, mid_y//2],dim = 1),axis=0))
        # TODO only take the center pixel of the output feature map to compute
        # FIXME replace = with append (in order to then compute the average over all layers?)

    return hook

def get_attention(attention):
    """Defines a hook for a transfomer architecture
    It should be registered on a layer such that the output contains the attention
    The model will then store the attentions in a list of atention for the batch of images
    This might be heavy and shall be modified @TODO improve memory performances
    """
    def hook(model, input, output):
        attention.append(output.cpu().detach().numpy())
    return hook