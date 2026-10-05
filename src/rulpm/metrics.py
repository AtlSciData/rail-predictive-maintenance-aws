import numpy as np
from sklearn.metrics import (average_precision_score, mean_squared_error, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def nasa_score(y_true, y_pred) -> float:
    """Asymmetric C-MAPSS score: late predictions (pred > true) are penalized more."""
    d = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1, np.exp(d / 10.0) - 1)))


def classification_report(y_true, proba, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true)
    out = {"recall": float(recall_score(y_true, proba >= threshold, zero_division=0))}
    if len(np.unique(y_true)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, proba))
        out["pr_auc"] = float(average_precision_score(y_true, proba))
    return out


def choose_threshold(y_true, proba, target_recall: float = 0.9) -> float:
    """Highest probability threshold whose recall on (validation) data is >= target_recall.

    The highest qualifying threshold keeps precision as large as possible while meeting the
    recall target. Select this on out-of-fold predictions, never on the test set.
    Falls back to the lowest threshold if the target recall is unreachable.
    """
    y_true = np.asarray(y_true)
    prec, rec, thr = precision_recall_curve(y_true, proba)
    ok = np.where(rec[:-1] >= target_recall)[0]
    return float(thr[ok].max()) if len(ok) else float(thr.min())


def threshold_report(y_true, proba, threshold: float) -> dict:
    y_true = np.asarray(y_true)
    pred = np.asarray(proba) >= threshold
    return {
        "threshold": float(threshold),
        "recall_at_thr": float(recall_score(y_true, pred, zero_division=0)),
        "precision_at_thr": float(precision_score(y_true, pred, zero_division=0)),
        "flagged": int(pred.sum()),
        "missed_failures": int(((y_true == 1) & ~pred).sum()),
        "false_alarms": int(((y_true == 0) & pred).sum()),
    }
