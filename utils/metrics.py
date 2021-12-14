"""
This file will contain the metrics of the framework
"""
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as mt
import wandb
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import RocCurveDisplay

class IOUMetric:
    """
    Class to calculate mean-iou using fast_hist method
    """

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    def add_batch(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())

    def evaluate(self):
        acc = np.diag(self.hist).sum() / self.hist.sum()
        acc_cls = np.diag(self.hist) / self.hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        mean_iu = np.nanmean(iu)
        freq = self.hist.sum(axis=1) / self.hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        return acc, acc_cls, iu, mean_iu, fwavacc


class AverageMeter:
    """
    Class to be an average meter for any average metric like loss, accuracy, etc..
    """

    def __init__(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.value = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    @property
    def val(self):
        return self.avg


class AverageMeterList:
    """
    Class to be an average meter for any average metric List structure like mean_iou_per_class
    """

    def __init__(self, num_cls):
        self.cls = num_cls
        self.value = [0] * self.cls
        self.avg = [0] * self.cls
        self.sum = [0] * self.cls
        self.count = [0] * self.cls
        self.reset()

    def reset(self):
        self.value = [0] * self.cls
        self.avg = [0] * self.cls
        self.sum = [0] * self.cls
        self.count = [0] * self.cls

    def update(self, val, n=1):
        for i in range(self.cls):
            self.value[i] = val[i]
            self.sum[i] += val[i] * n
            self.count[i] += n
            self.avg[i] = self.sum[i] / self.count[i]

    @property
    def val(self):
        return self.avg


def cls_accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    view = target.view(1, -1).expand_as(pred)
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k / batch_size)
    return res

def consusion_matrix(output,target):
    from sklearn.metrics import ConfusionMatrixDisplay
    ConfusionMatrixDisplay.from_predictions(target.astype(np.int32), output.astype(np.int32))
    plot = wandb.Image(plt)
    plt.close()
    return plot
    

def multi_cls_accuracy(output, target):
    plt.ioff()
    p,r,f,_ = mt.precision_recall_fscore_support(target, output, average='weighted')
    plot = wandb.Image(plt)
    plt.close()
    return p,r,f,plot

def compute_metrics(multi_output, multi_target,num_classes):
    onehot_encoder = OneHotEncoder(sparse=False)
    onehot_encoder.fit(np.arange(num_classes).reshape(-1,1))
    output = onehot_encoder.transform(multi_output.reshape(-1,1))
    target = onehot_encoder.transform(multi_target.reshape(-1,1))
    ap = mt.average_precision_score(target, output,average="weighted")
    f1 = mt.f1_score(target, output, average='weighted',zero_division=1)
    pr = mt.precision_score(target, output, average='weighted',zero_division=1)
    rc = mt.recall_score(target, output, average='weighted',zero_division=1)


    # plt.ioff()
    # plt.figure()
    # plt.title('Receiver operating characteristic weighted')
    # plt.legend(loc="lower right")

    dic = {"epoch/Average Precision (weighted)"  : ap,
           "epoch/F1 score (weighted) "          : f1,
           "epoch/Precision (weighted)"          : pr,
           "epoch/Recall (weighted)"             : rc
        }
           # "epoch/Roc (weighted)"                : wandb.Image(plt)}
    # plt.close()
    return dic

def multi_cls_roc(target, output,num_classes):
    from sklearn.preprocessing import label_binarize
    target = label_binarize(target, classes=np.arange(num_classes))
    output = label_binarize(output, classes=np.arange(num_classes))
    
    import matplotlib.pyplot as plt
    from itertools import cycle
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import average_precision_score

    
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(num_classes):
        precision[i], recall[i], _ = precision_recall_curve(target[:, i], output[:, i])
        average_precision[i] = average_precision_score(target[:, i], output[:, i])

    # A "weighted-average": quantifying score on all classes jointly
    precision["weighted"], recall["weighted"], _ = precision_recall_curve(
        target.ravel(), output.ravel()
    )
    average_precision["weighted"] = average_precision_score(target, output, average="weighted")
    plt.ioff()
    plt.figure()
    
    display = mt.PrecisionRecallDisplay(
    recall=recall["weighted"],
    precision=precision["weighted"],
    average_precision=average_precision["weighted"],
    )
    display.plot()
    _ = display.ax_.set_title("weighted-averaged over all classes")


    # setup plot details
    colors = cycle(["navy", "turquoise", "darkorange", "cornflowerblue", "teal"])

    _, ax = plt.subplots(figsize=(7, 8))

    # number of isoline to plot 
    num_iso = 6
    f_scores = np.linspace(0.2, 0.8, num=num_iso)
    lines, labels = [], []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
        plt.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))

    display = mt.PrecisionRecallDisplay(
        recall=recall["weighted"],
        precision=precision["weighted"],
        average_precision=average_precision["weighted"],
    )
    display.plot(ax=ax, name="weighted-average precision-recall", color="gold")

    for i, color in zip(range(num_classes), colors):
        display = mt.PrecisionRecallDisplay(
            recall=recall[i],
            precision=precision[i],
            average_precision=average_precision[i],
        )
        display.plot(ax=ax, name=f"Precision-recall for class {i}", color=color)

    # add the legend for the iso-f1 curves
    handles, labels = display.ax_.get_legend_handles_labels()
    handles.extend([l])
    labels.extend(["iso-f1 curves"])
    # set the legend and the axes
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(handles=handles, labels=labels, loc="best")
    ax.set_title("Extension of Precision-Recall curve to multi-class")
    wandb_plot = wandb.Image(plt)
    plt.close()
    return wandb_plot,np.mean(average_precision["weighted"])