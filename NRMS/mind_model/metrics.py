from sklearn.metrics import roc_auc_score
import numpy as np

def dcg_score(y_true, y_score, k=10):
    """Computing dcg score metric at k.

        Args:
            y_true (np.ndarray): Ground-truth labels.
            y_score (np.ndarray): Predicted labels.

        Returns:
            np.ndarray: dcg scores.
        """
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2**y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10):
    """Computing ndcg score metric at k.

        Args:
            y_true (np.ndarray): Ground-truth labels.
            y_score (np.ndarray): Predicted labels.

        Returns:
            numpy.ndarray: ndcg scores.
        """
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def mrr_score(y_true, y_score):
    """Computing mrr score metric.

        Args:
            y_true (np.ndarray): Ground-truth labels.
            y_score (np.ndarray): Predicted labels.

        Returns:
            numpy.ndarray: mrr scores.
        """
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)



def ctr_score(y_true, y_score, k=1):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    return np.mean(y_true)
