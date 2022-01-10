import importlib

from torch import nn, optim
from torch.nn import MarginRankingLoss



def get_net(arch, network_param, optimizer_param = None, loss_param = None):
    """
    Get Network Architecture based on arguments provided
    """
    # FIXME this iss fucking strange the import needs to be done twice to work
    # TODO dictionnary for parameters (network, optim, loss)
    mod = importlib.import_module(f"models.{arch}")
    net = getattr(mod, arch)
    if optimizer_param is not None and loss_param is not None:
        return net(network_param, optimizer_param, loss_param)
    elif optimizer_param is not None:
        return net(network_param, optimizer_param)
    elif loss_param is not None:
        return net(network_param, loss_param)
    else : 
        return net(network_param)


def get_datamodule(datamodule,data_param,dataset = None):
    """
    Fetch Network Function Pointer
    """
    module = "datamodules." + datamodule
    mod = importlib.import_module(module)
    net = getattr(mod, datamodule)
    return net(data_param,dataset)



def import_class(name, instantiate=None):

    namesplit = name.split(".")
    module = importlib.import_module(".".join(namesplit[:-1]))
    imported_class = getattr(module, namesplit[-1])

    if imported_class:
        if instantiate is not None:
            return imported_class(**instantiate)
        else:
            return imported_class
    raise Exception("Class {} can be imported".format(import_class))
