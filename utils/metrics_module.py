from utils.agent_utils import import_class
from utils.constant import PASCAL_VOC_classes
import wandb
import numpy as np
import torch
"""
https://torchmetrics.readthedocs.io/en/stable/references/modules.html#base-class MODULE METRICS
"""

class MetricsModule():

    def __init__(self, set_name, params,device,) -> None:
        """
        metrics : list of name metrics e.g ["Accuracy", "IoU"]
        set_name: val/train/test
        """
        self.device = device
        dict_metrics = {}
        if set_name != "train":
            for name in params.metrics:
                instance = import_class("torchmetrics." + name)(compute_on_step=False, **params.pop("metrics"))
                dict_metrics[name.lower()] = instance.to(device)
        else:
            dict_metrics["iou"] = import_class(
                "torchmetrics.IoU")(compute_on_step=False, num_classes=params.num_classes).to(device)

        self.dict_metrics = dict_metrics

    def update_metrics(self, x, y):
        # TODO void class ne doit pas intervenir lors du calcul des metrics et de la loss

        for k, m in self.dict_metrics.items():

            if k != "averageprecision":
                preds = torch.argmax(x, dim=1)
            else:
                preds = x

            # metric on current batch
            m(preds, y)  # update metrics (torchmetrics method)

    def log_metrics(self, name, pl_module):

        for k, m in self.dict_metrics.items():

            # metric on all batches using custom accumulation
            metric = m.compute()

            if k =="confusionmatrix":
                class_names = [v for k, v in PASCAL_VOC_classes.items()]
                class_names.remove("void")
                wandb.log({name + "confusionmatrix" : wandb.plots.HeatMap(class_names, class_names, metric.cpu(), show_text=True)})
            else:
                if metric.numel() != 1:
                    data = [[PASCAL_VOC_classes[idx], val]
                            for idx, val in enumerate(metric)]
                    table = wandb.Table(columns=["label", "value"], data=data)
                    wandb.log({name + k: table})
                else:
                    pl_module.log(name + k, metric)

            # Reseting internal state such that metric ready for new data
            m.reset()
            m.to(self.device)
