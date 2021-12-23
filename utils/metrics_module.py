from utils.agent_utils import import_class
from utils.constant import PASCAL_VOC_classes
import wandb

class MetricsModule():

    def __init__(self, config_metrics) -> None:
        """
        metrics : list of name metrics e.g ["Accuracy", "IoU"]
        """
        list_metrics = []
        name_metrics = []
        for name, params in config_metrics.items():
            if name != "ConfusionMatrix":
                list_metrics.append(import_class("torchmetrics." + name)
                                    (**params)
                                    )
            else:
                list_metrics.append("ConfusionMatrix")
            name_metrics.append(name.lower())

        self.inst_metrics = list_metrics
        self.name_metrics = name_metrics

    def compute_metrics(self, x, y):
        # TODO void class ne doit pas intervenir lors du calcul des metrics

        dic_metrics = {}
        for idx, m in enumerate(self.inst_metrics):
            if self.name_metrics[idx] == "confusionmatrix":
                class_names = [v for k, v in PASCAL_VOC_classes.items()]
                cm = wandb.plot.confusion_matrix(probs=None,
                                                y_true=y.numpy().flatten(), preds=x.numpy().flatten(),
                                                class_names=class_names)
                dic_metrics[self.name_metrics[idx]] = cm
            else:
                dic_metrics[self.name_metrics[idx]] = m(x, y)

        self.metrics = dic_metrics

    def log_metrics(self, name, logger):

        for k, v in self.metrics.items():

            if k == "confusionmatrix":
                wandb.log({name + k: v})
            else:
                if v.numel() != 1:
                    data = [[PASCAL_VOC_classes[idx], val]
                            for idx, val in enumerate(v)]
                    table = wandb.Table(columns=["label", "values"], data=data)
                    wandb.log(
                        {name + k: wandb.plot.bar(table, "label", "value")})
                else:
                    logger.log(name + k, v)
