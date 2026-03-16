import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier, AdaBoostRegressor
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.model_selection import StratifiedKFold, KFold, GridSearchCV, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pickle

from app.stats import class_scoring, regress_scoring, get_class_stats, get_regress_stats

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


def build_qsar_model(X: pd.DataFrame, y: pd.Series, alg: str, scale=True):
    """Build a classification QSAR model"""
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
    binary_preds = (probs >= 0.5).astype(int)

    five_fold_stats = get_class_stats(None, y, probs)
    final_cv_predictions = pd.concat([probs.rename("Probability"), binary_preds.rename("Prediction")], axis=1)
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
    five_fold_stats = get_regress_stats(None, y, predictions)

    regressive_model = pickle.dumps(best_estimator)
    return regressive_model, predictions.rename("Prediction"), five_fold_stats


if __name__ == '__main__':
    print(CLASSIFIER_ALGORITHMS_DICT)