import torch
def get_activation(name, features):
    def hook(model, input, output):
        if output.requires_grad:
            mid = output.shape[-1]
            features[name].append(torch.mean(output[:, :, mid//2, mid//2]))
        # TODO only take the center pixel of the output feature map to compute
        # FIXME replace = with append (in order to then compute the average over all layers?)

    return hook
