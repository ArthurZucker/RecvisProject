from utils.agent_utils import import_class
from utils.constant import PASCAL_VOC_classes
import wandb
import numpy as np
import torch
"""
https://torchmetrics.readthedocs.io/en/stable/references/modules.html#base-class MODULE METRICS
"""

class MetricsModule():

    def __init__(self, set_name, config_metrics, n_classes=None) -> None:
        """
        metrics : list of name metrics e.g ["Accuracy", "IoU"]
        set_name: val/train/test
        """
        dict_metrics = {}
        if set_name != "train":
            for name, params in config_metrics.items():
                if name != "ConfusionMatrix":
                    instance = import_class("torchmetrics." + name)(**params)
                    dict_metrics[name.lower()] = instance
                else:
                    dict_metrics["confusionmatrix"] = (
                        np.zeros(1), np.zeros(1))
        else:
            dict_metrics["iou"] = import_class(
                "torchmetrics.IoU")(num_classes=n_classes)

        self.dict_metrics = dict_metrics

    def update_metrics(self, x, y):
        # TODO void class ne doit pas intervenir lors du calcul des metrics et de la loss

        for k, m in self.dict_metrics.items():

            if k != "averageprecision":
                preds = torch.argmax(x, dim=1)
            else:
                preds = x

            if k == "confusionmatrix":
                self.dict_metrics[k] = (np.concatenate(
                    (m[0], preds.numpy().flatten())),
                    np.concatenate((m[1], y.numpy().flatten())))
            else:
                # metric on current batch
                m(preds, y)  # update metrics (torchmetrics method)

    def log_metrics(self, name, logger):

        for k, m in self.dict_metrics.items():

            if k == "confusionmatrix":
                class_names = [v for k, v in PASCAL_VOC_classes.items()]
                preds = np.delete(m[0], 0)
                labels = np.delete(m[1], 0)
                cm = wandb.plot.confusion_matrix(probs=None,
                                                y_true=labels, preds=preds,
                                                class_names=class_names)
                wandb.log({name + k: cm})

                self.dict_metrics["confusionmatrix"] = (
                    np.zeros(1), np.zeros(1))  # reset
            else:
                # metric on all batches using custom accumulation
                metric = m.compute()
                if metric.numel() != 1:
                    data = [[PASCAL_VOC_classes[idx], val]
                            for idx, val in enumerate(metric)]
                    table = wandb.Table(columns=["label", "value"], data=data)
                    wandb.log({name + k: table})
                else:
                    logger.log_metrics({name + k: metric})

                # Reseting internal state such that metric ready for new data
                m.reset()
