from sklearn.metrics import cohen_kappa_score, matthews_corrcoef, precision_score, recall_score, confusion_matrix
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score, roc_curve, auc, f1_score, accuracy_score, r2_score, \
    max_error, \
    mean_squared_error, mean_absolute_percentage_error, d2_pinball_score, explained_variance_score
import numpy as np


def get_class_stats(model, X, y):
    """
    Evaluate classification model or prediction vector.
    If model is None, assume X is true labels and y is probabilities.
    """
    if model is None:
        predicted_probas = y
        predicted_classes = (predicted_probas >= 0.5).astype(int)
        y = X
    else:
        predicted_probas = model.predict_proba(X)[:, 1]
        predicted_classes = model.predict(X)

    predicted_probas = np.nan_to_num(predicted_probas, nan=0.0, posinf=1.0)

    acc = accuracy_score(y, predicted_classes)
    f1_sc = f1_score(y, predicted_classes)
    fpr_tr, tpr_tr, _ = roc_curve(y, predicted_probas)
    roc_auc = auc(fpr_tr, tpr_tr)

    cohen_kappa = cohen_kappa_score(y, predicted_classes)
    matthews_corr = matthews_corrcoef(y, predicted_classes)
    precision = precision_score(y, predicted_classes)
    recall = recall_score(y, predicted_classes)

    cm = confusion_matrix(y, predicted_classes)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ccr = (recall + specificity) / 2
    else:
        specificity = None
        ccr = None

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
        'R2-score': None,
        'Explained-variance': None,
        'Max-error': None,
        'Mean-squared-error': None,
        'Mean-absolute-percentage-error': None,
        'D2-pinball-score': None
    }


class_scoring = {
    'ACC': make_scorer(accuracy_score),
    'F1-Score': make_scorer(f1_score),
    'AUC': make_scorer(roc_auc_score, needs_proba=True),
    'Cohen\'s Kappa': make_scorer(cohen_kappa_score),
    'MCC': make_scorer(matthews_corrcoef),
    'Precision': make_scorer(precision_score),
    'Recall': make_scorer(recall_score)
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
        'R2-score': r2,
        'Explained-variance': explained_var,
        'Max-error': m_error,
        'Mean-squared-error': ms_error,
        'Mean-absolute-percentage-error': mape,
        'D2-pinball-score': pinball_score
    }


regress_scoring = {
    'R2-score': make_scorer(r2_score),
    'Explained-variance': make_scorer(explained_variance_score),
    'Max-error': make_scorer(max_error, greater_is_better=False),
    'Mean-squared-error': make_scorer(mean_squared_error, greater_is_better=False),
    'Mean-absolute-percentage-error': make_scorer(mean_absolute_percentage_error, greater_is_better=False),
    'D2-pinball-score': make_scorer(d2_pinball_score)  # This is a utility score (higher is better)
}