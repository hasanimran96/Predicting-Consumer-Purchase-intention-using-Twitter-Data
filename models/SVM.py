from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    recall_score,
    precision_score,
)

kfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

np.random.seed(1)

pipeline_svm = make_pipeline(
    vectorizer, SVC(probability=True, kernel="linear", class_weight="balanced")
)

grid_svm = GridSearchCV(
    pipeline_svm,
    param_grid={"svc__C": [0.01, 0.1, 1]},
    cv=kfolds,
    scoring="roc_auc",
    verbose=1,
    n_jobs=-1,
)

grid_svm.fit(X_train, y_train)
grid_svm.score(X_test, y_test)

grid_svm.best_params_
grid_svm.best_score_


def report_results(model, X, y):
    pred_proba = model.predict_proba(X)[:, 1]
    pred = model.predict(X)

    auc = roc_auc_score(y, pred_proba)
    acc = accuracy_score(y, pred)
    f1 = f1_score(y, pred)
    prec = precision_score(y, pred)
    rec = recall_score(y, pred)
    result = {"auc": auc, "f1": f1, "acc": acc, "precision": prec, "recall": rec}
    return result


report_results(grid_svm.best_estimator_, X_test, y_test)
