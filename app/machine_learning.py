import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier, AdaBoostRegressor
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.model_selection import StratifiedKFold, KFold, GridSearchCV, cross_val_predict
from sklearn.metrics import roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pickle

from app.stats import class_scoring, regress_scoring, get_class_stats_by_fold, get_regress_stats_by_fold

seed = 0

CLASSIFIER_ALGORITHMS = [
    ('RF', RandomForestClassifier(max_depth=10, class_weight='balanced', random_state=seed),
     {'RF__n_estimators': [5, 10, 25, 100, 200]}),
    ('kNN', KNeighborsClassifier(metric='euclidean'),
     {'kNN__n_neighbors': [1, 3, 5, 10, 20], 'kNN__weights': ['uniform', 'distance']}),
    ('SVM', SVC(probability=True, class_weight='balanced', random_state=seed),
     {'SVM__kernel': ['linear', 'rbf', 'poly'], 'SVM__gamma': [1e-2, 1e-3], 'SVM__C': [0.1, 1, 10]}),
    ('BNB', BernoulliNB(alpha=1.0), {}),
    ('ADA', AdaBoostClassifier(n_estimators=100, learning_rate=0.9, random_state=seed), {})
]

CLASSIFIER_ALGORITHMS_DICT = {name: (name, model, params) for name, model, params in CLASSIFIER_ALGORITHMS}

REGRESSOR_ALGORITHMS = [
    ('RF', RandomForestRegressor(max_depth=10, random_state=seed),
     {'RF__n_estimators': [5, 10, 25, 100, 200]}),
    ('kNN', KNeighborsRegressor(metric='euclidean'),
     {'kNN__n_neighbors': [1, 3, 5, 10, 20], 'kNN__weights': ['uniform', 'distance']}),
    ('SVM', SVR(), {'SVM__kernel': ['linear', 'rbf', 'poly'], 'SVM__gamma': [1e-2, 1e-3], 'SVM__C': [0.1, 1, 10]}),
    ('ADA', AdaBoostRegressor(n_estimators=100, learning_rate=0.9, random_state=seed), {})
]

REGRESSOR_ALGORITHMS_DICT = {name: (name, model, params) for name, model, params in REGRESSOR_ALGORITHMS}


def _youden_threshold(y_true, probs):
    """
    Select a classification threshold using Youden's J statistic.

    This function uses already-computed cross-validated out-of-fold
    probabilities, so it does not require extra model fitting.
    Falls back to 0.5 if the threshold cannot be calculated.
    """
    y_true = pd.Series(y_true)
    probs = pd.Series(probs, index=y_true.index)

    if y_true.nunique(dropna=True) < 2:
        return 0.5

    fpr, tpr, thresholds = roc_curve(y_true, probs)
    youden_j = tpr - fpr

    valid = np.isfinite(thresholds)
    if not valid.any():
        return 0.5

    youden_j = np.where(valid, youden_j, np.nan)
    best_idx = int(np.nanargmax(youden_j))
    threshold = float(thresholds[best_idx])

    if not np.isfinite(threshold):
        return 0.5

    return float(np.clip(threshold, 0.0, 1.0))


def _class_balance_ratio(y):
    """
    Return the minority/majority class ratio.

    Examples:
        50 active, 950 inactive  -> 50/950 = 0.053
        450 active, 550 inactive -> 450/550 = 0.818

    A lower value means stronger class imbalance.
    """
    counts = pd.Series(y).value_counts(dropna=True)

    if len(counts) < 2:
        return None

    return float(counts.min() / counts.max())


def _auto_threshold_from_training_data(y, probs, imbalance_ratio_cutoff=0.5):
    """
    Automatically choose the classification threshold from the training data.

    Rule:
        minority/majority ratio < imbalance_ratio_cutoff
            -> imbalanced endpoint, use Youden's J threshold
        minority/majority ratio >= imbalance_ratio_cutoff
            -> balanced endpoint, keep standard 0.5 threshold

    With the default cutoff of 0.5, class ratios worse than 1:2 are treated
    as imbalanced.
    """
    balance_ratio = _class_balance_ratio(y)

    if balance_ratio is not None and balance_ratio < imbalance_ratio_cutoff:
        threshold = _youden_threshold(y, probs)
        threshold_method_used = "auto_youden_for_imbalanced_endpoint"
    else:
        threshold = 0.5
        threshold_method_used = "auto_fixed_0.5_for_balanced_endpoint"

    return threshold, threshold_method_used, balance_ratio


def build_qsar_model(
    X: pd.DataFrame,
    y: pd.Series,
    alg: str,
    scale=True,
    imbalance_ratio_cutoff=0.5
):
    """
    Build a classification QSAR model.

    The threshold is selected automatically from the training data:
        - balanced endpoints keep the standard 0.5 probability cutoff
        - imbalanced endpoints use Youden's J threshold calculated from
          cross-validated out-of-fold probabilities

    Users do not need to select the threshold method.
    """
    if alg not in CLASSIFIER_ALGORITHMS_DICT:
        raise ValueError(f"Unknown classification algorithm: {alg}")

    cv = StratifiedKFold(shuffle=True, n_splits=5, random_state=seed)
    name, model, params = CLASSIFIER_ALGORITHMS_DICT[alg]

    pipe = Pipeline([('scaler', StandardScaler()), (name, model)]) if scale else Pipeline([(name, model)])

    grid_search = GridSearchCV(pipe, param_grid=params, cv=cv, scoring=class_scoring, refit='AUC')
    grid_search.fit(X, y)
    best_estimator = grid_search.best_estimator_

    cv_predictions = pd.DataFrame(
        cross_val_predict(best_estimator, X, y, cv=cv, method='predict_proba'),
        index=y.index
    )

    probs = cv_predictions.iloc[:, 1]

    threshold, threshold_method_used, balance_ratio = _auto_threshold_from_training_data(
        y,
        probs,
        imbalance_ratio_cutoff=imbalance_ratio_cutoff
    )

    binary_preds = (probs >= threshold).astype(int)

    fold_ids = pd.Series(index=y.index, dtype=int)
    for fold_idx, (_, test_idx) in enumerate(cv.split(X, y)):
        fold_ids.iloc[test_idx] = fold_idx

    five_fold_stats = get_class_stats_by_fold(y, probs, fold_ids, threshold=threshold)

    # Store the threshold used for reporting and QSAR Predictor.
    # Extra fields are useful for debugging/logging and harmless if not saved to DB.
    five_fold_stats['Threshold'] = threshold
    five_fold_stats['Threshold_Method'] = threshold_method_used
    five_fold_stats['Class_Balance_Ratio'] = balance_ratio
    five_fold_stats['Imbalance_Ratio_Cutoff'] = imbalance_ratio_cutoff

    final_cv_predictions = pd.concat([
        probs.rename("Probability"),
        pd.Series(threshold, index=y.index, name="Threshold"),
        binary_preds.rename("Prediction")
    ], axis=1)

    binary_model = pickle.dumps(best_estimator)

    return binary_model, final_cv_predictions, five_fold_stats

def build_qsar_model_regression(X: pd.DataFrame, y: pd.Series, alg: str, scale=True):
    """Build a regression QSAR model"""
    if alg not in REGRESSOR_ALGORITHMS_DICT:
        raise ValueError(f"Unknown regression algorithm: {alg}")

    cv = KFold(shuffle=True, n_splits=5, random_state=seed)
    name, model, params = REGRESSOR_ALGORITHMS_DICT[alg]

    pipe = Pipeline([('scaler', StandardScaler()), (name, model)]) if scale else Pipeline([(name, model)])

    grid_search = GridSearchCV(pipe, param_grid=params, cv=cv, scoring=regress_scoring,
                               refit='R2-score')
    grid_search.fit(X, y)
    best_estimator = grid_search.best_estimator_

    predictions = pd.Series(cross_val_predict(best_estimator, X, y, cv=cv), index=y.index)

    fold_ids = pd.Series(index=y.index, dtype=int)
    for fold_idx, (_, test_idx) in enumerate(cv.split(X, y)):
        fold_ids.iloc[test_idx] = fold_idx

    five_fold_stats = get_regress_stats_by_fold(y, predictions, fold_ids)

    regressive_model = pickle.dumps(best_estimator)
    return regressive_model, predictions.rename("Prediction"), five_fold_stats


if __name__ == '__main__':
    print(CLASSIFIER_ALGORITHMS_DICT)