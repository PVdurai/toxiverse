from sklearn.metrics import cohen_kappa_score, matthews_corrcoef, precision_score, recall_score, confusion_matrix
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score, roc_curve, auc, f1_score, accuracy_score, r2_score, \
    max_error, \
    mean_squared_error, mean_absolute_percentage_error, d2_pinball_score, explained_variance_score
import numpy as np
import pandas as pd


def _mean(values):
    """Mean across CV folds, ignoring metrics that are undefined for a fold."""
    values = pd.Series(values, dtype="float64").dropna()
    if len(values) == 0:
        return None
    return float(values.mean())


def _std(values):
    """Sample SD across CV folds, ignoring metrics that are undefined for a fold."""
    values = pd.Series(values, dtype="float64").dropna()
    if len(values) <= 1:
        return 0.0
    return float(values.std(ddof=1))


def get_class_stats(model, X, y, threshold=0.5):
    """
    Evaluate classification model or prediction vector.
    If model is None, assume X is true labels and y is probabilities.
    threshold is used only when converting probabilities to binary classes.
    """
    if model is None:
        predicted_probas = y
        predicted_classes = (predicted_probas >= threshold).astype(int)
        y = X
    else:
        predicted_probas = model.predict_proba(X)[:, 1]
        predicted_classes = model.predict(X)

    y = np.asarray(y)
    predicted_probas = np.nan_to_num(predicted_probas, nan=0.0, posinf=1.0, neginf=0.0)

    acc = accuracy_score(y, predicted_classes)
    f1_sc = f1_score(y, predicted_classes, zero_division=0)

    has_two_classes = len(np.unique(y)) == 2
    if has_two_classes:
        fpr_tr, tpr_tr, _ = roc_curve(y, predicted_probas)
        roc_auc = auc(fpr_tr, tpr_tr)
        cohen_kappa = cohen_kappa_score(y, predicted_classes)
        matthews_corr = matthews_corrcoef(y, predicted_classes)
    else:
        roc_auc = np.nan
        cohen_kappa = np.nan
        matthews_corr = np.nan
    precision = precision_score(y, predicted_classes, zero_division=0)
    recall = recall_score(y, predicted_classes, zero_division=0)

    cm = confusion_matrix(y, predicted_classes, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ccr = (recall + specificity) / 2

    return {
        'ACC': acc,
        'F1-Score': f1_sc,
        'AUC': roc_auc,
        'Cohen\'s Kappa': cohen_kappa,
        'MCC': matthews_corr,
        'Precision': precision,
        'Recall': recall,
        'Specificity': specificity,
        'CCR': ccr,
        'Threshold': threshold,
        'ACC_SD': None,
        'F1-Score_SD': None,
        'AUC_SD': None,
        'Cohen\'s Kappa_SD': None,
        'MCC_SD': None,
        'Precision_SD': None,
        'Recall_SD': None,
        'Specificity_SD': None,
        'CCR_SD': None,
        'R2-score': None,
        'Explained-variance': None,
        'Max-error': None,
        'Mean-squared-error': None,
        'Mean-absolute-percentage-error': None,
        'D2-pinball-score': None,
        'R2-score_SD': None,
        'Explained-variance_SD': None,
        'Max-error_SD': None,
        'Mean-squared-error_SD': None,
        'Mean-absolute-percentage-error_SD': None,
        'D2-pinball-score_SD': None
    }


def get_class_stats_by_fold(y_true, predicted_probas, fold_ids, threshold=0.5):
    """
    Report classification metrics as mean +/- SD across held-out CV folds.
    Uses already-computed out-of-fold probabilities, so this adds negligible cost.
    """
    y_true = pd.Series(y_true)
    predicted_probas = pd.Series(predicted_probas, index=y_true.index)
    fold_ids = pd.Series(fold_ids, index=y_true.index)

    fold_stats = []
    for fold in sorted(fold_ids.dropna().unique()):
        mask = fold_ids == fold
        fold_stats.append(get_class_stats(None, y_true.loc[mask], predicted_probas.loc[mask], threshold=threshold))

    mean_stats = get_class_stats(None, y_true, predicted_probas, threshold=threshold)
    metrics = ['ACC', 'F1-Score', 'AUC', "Cohen's Kappa", 'MCC', 'Precision', 'Recall', 'Specificity', 'CCR']
    for metric in metrics:
        fold_values = [stats[metric] for stats in fold_stats]
        mean_stats[metric] = _mean(fold_values)
        mean_stats[f'{metric}_SD'] = _std(fold_values)
    mean_stats['Threshold'] = threshold
    return mean_stats


class_scoring = {
    'ACC': make_scorer(accuracy_score),
    'F1-Score': make_scorer(f1_score),
    'AUC': make_scorer(roc_auc_score, needs_proba=True),
    'Cohen\'s Kappa': make_scorer(cohen_kappa_score),
    'MCC': make_scorer(matthews_corrcoef),
    'Precision': make_scorer(precision_score, zero_division=0),
    'Recall': make_scorer(recall_score, zero_division=0)
}


def get_regress_stats(model, X, y):
    """
    Evaluate regression model or prediction vector.
    If model is None, assume X is true values and y is predicted.
    """
    if model is None:
        predicted_values = y
        y = X
    else:
        predicted_values = model.predict(X)

    r2 = r2_score(y, predicted_values)
    explained_var = explained_variance_score(y, predicted_values)
    m_error = max_error(y, predicted_values)
    ms_error = mean_squared_error(y, predicted_values)
    mape = mean_absolute_percentage_error(y, predicted_values)
    pinball_score = d2_pinball_score(y, predicted_values)

    return {
        'ACC': None,
        'F1-Score': None,
        'AUC': None,
        'Cohen\'s Kappa': None,
        'MCC': None,
        'Precision': None,
        'Recall': None,
        'Specificity': None,
        'CCR': None,
        'Threshold': None,
        'ACC_SD': None,
        'F1-Score_SD': None,
        'AUC_SD': None,
        'Cohen\'s Kappa_SD': None,
        'MCC_SD': None,
        'Precision_SD': None,
        'Recall_SD': None,
        'Specificity_SD': None,
        'CCR_SD': None,
        'R2-score': r2,
        'Explained-variance': explained_var,
        'Max-error': m_error,
        'Mean-squared-error': ms_error,
        'Mean-absolute-percentage-error': mape,
        'D2-pinball-score': pinball_score,
        'R2-score_SD': None,
        'Explained-variance_SD': None,
        'Max-error_SD': None,
        'Mean-squared-error_SD': None,
        'Mean-absolute-percentage-error_SD': None,
        'D2-pinball-score_SD': None
    }


def get_regress_stats_by_fold(y_true, predictions, fold_ids):
    """Report regression metrics as mean +/- SD across held-out CV folds."""
    y_true = pd.Series(y_true)
    predictions = pd.Series(predictions, index=y_true.index)
    fold_ids = pd.Series(fold_ids, index=y_true.index)

    fold_stats = []
    for fold in sorted(fold_ids.dropna().unique()):
        mask = fold_ids == fold
        fold_stats.append(get_regress_stats(None, y_true.loc[mask], predictions.loc[mask]))

    mean_stats = get_regress_stats(None, y_true, predictions)
    metrics = ['R2-score', 'Explained-variance', 'Max-error', 'Mean-squared-error',
               'Mean-absolute-percentage-error', 'D2-pinball-score']
    for metric in metrics:
        fold_values = [stats[metric] for stats in fold_stats]
        mean_stats[metric] = _mean(fold_values)
        mean_stats[f'{metric}_SD'] = _std(fold_values)
    return mean_stats


regress_scoring = {
    'R2-score': make_scorer(r2_score),
    'Explained-variance': make_scorer(explained_variance_score),
    'Max-error': make_scorer(max_error, greater_is_better=False),
    'Mean-squared-error': make_scorer(mean_squared_error, greater_is_better=False),
    'Mean-absolute-percentage-error': make_scorer(mean_absolute_percentage_error, greater_is_better=False),
    'D2-pinball-score': make_scorer(d2_pinball_score)  # This is a utility score (higher is better)
}
