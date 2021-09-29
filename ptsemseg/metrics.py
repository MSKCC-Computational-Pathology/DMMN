import numpy as np


class runningScore(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask],
            minlength=n_class ** 2,
        ).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(
                lt.flatten(), lp.flatten(), self.n_classes
            )

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        hist = hist[1:, 1:]  # exclude non-labeled regions
        # print(hist)
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = 2*np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0)) # dice coefficient
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))
        
        

        recalls = {'recall_{}'.format(cls + 1): recall
                 for cls, recall in enumerate(calculate_recall(hist))}
        precisions = {'precision_{}'.format(cls + 1): precision
                 for cls, precision in enumerate(calculate_precision(hist))}
        average_recall = np.mean(list(recalls.values()))
        average_precision = np.mean(list(precisions.values()))

        return (
            {
                "Overall Acc: \t": acc,
                "Mean Acc : \t": acc_cls,
                "FreqW Acc : \t": fwavacc,
                "Mean IoU : \t": mean_iu,
            },
            cls_iu, hist, mean_iu, recalls, precisions, average_recall, average_precision
        )

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))



def calculate_recall(confusion_matrix):
    recalls = []
    for index in range(confusion_matrix.shape[0]):
        true_positives = confusion_matrix[index, index]
        false_negatives = confusion_matrix[index, :].sum() - true_positives
        denom = true_positives + false_negatives
        if denom == 0:
            recall = 0
        else:
            recall = float(true_positives) / denom
        recalls.append(recall)
    return recalls

def calculate_precision(confusion_matrix):
    precisions = []
    for index in range(confusion_matrix.shape[0]):
        true_positives = confusion_matrix[index, index]
        false_positives = confusion_matrix[:, index].sum() - true_positives
        denom = true_positives + false_positives
        if denom == 0:
            precision = 0
        else:
            precision = float(true_positives) / denom
        precisions.append(precision)
    return precisions
    
class averageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

