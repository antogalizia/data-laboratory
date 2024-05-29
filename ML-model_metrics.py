# Evaluation metrics for ML models.

def binary_confusion_matrix(y_test, y_pred):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(len(y_test)):
        if y_test[i]:
            if y_pred[i]:
                tp += 1
            else:
                fn += 1
        else:
            if y_pred[i]:
                fp += 1
            else:
                tn += 1

    return tp, tn, fp, fn


def accuracy_score(tp, tn, fp, fn):
    """
    :param tp: Number of true positives instances
    :param tn: Number of true negatives instances
    :param fp: Number of false positives instances
    :param fn: Number of false negatives instancias
    :return: ML Model accuracy score. Percentage of data training correctly classified.
    """

    acc = (tp + tn)/(tp + tn + fp + fn)
    return acc


def precision_score(tp, fp):
    """
    :param tp: Number of true positives instances
    :param fp: Number of false positives instances
    :return: ML Model precision score. From positives instances how man
    """

    prec = tp/(tp + fp)
    return prec


def recall_score(tp, fn):
    """
    :param tp: Number of true positives instances
    :param fn: Number of false negatives instances
    :return: ML Model completeness score
    """

    rec = tp/(tp + fn)
    return rec

def f1_score(tp, fp, fn):
    prec = precision_score(tp, fp)
    rec = recall_score(tp, fn)
    f1 = 2*prec*rec/(prec+rec)
    return f1



