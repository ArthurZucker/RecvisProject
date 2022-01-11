import importlib

from torch import nn, optim
from torch.nn import MarginRankingLoss
from easydict import EasyDict


def get_net(arch, network_param):
    """
    Get Network Architecture based on arguments provided
    """
    mod = importlib.import_module(f"models.{arch}")
    net = getattr(mod, arch)
    return net(network_param)


def get_lightning_module(arch, config):

    mod = importlib.import_module(f"lightningmodules.{arch}")
    net = getattr(mod, arch)
    return net(EasyDict(config))


def get_datamodule(datamodule, data_param, dataset = None):
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
