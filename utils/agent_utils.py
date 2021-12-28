import importlib

from torch import nn, optim
from torch.nn import MarginRankingLoss


def get_net(args, name=None):
    """
    Get Network Architecture based on arguments provided
    """
    name = name if name is not None else args.arch
    module = "models." + name
    mod = importlib.import_module(module)
    net = getattr(mod, name)
    return net(args)


def get_datamodule(args):
    """
    Fetch Network Function Pointer
    """
    module = "datamodule." + args.datamodule
    mod = importlib.import_module(module)
    net = getattr(mod, (args.datamodule))
    return net(args)



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
