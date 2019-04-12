import pandas as pd
import Clean as cl
from sklearn import svm

# from sklearn.cross_validation import train_test_split
from sklearn import model_selection, naive_bayes
import numpy as np

from sklearn.metrics import make_scorer, accuracy_score, f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    recall_score,
    precision_score,
)
from sklearn.metrics import confusion_matrix

# ------------------------
# Path declaration
# path = "..\data\AnnotatedData3.csv"  # windows
path = "data/AnnotatedData3.csv"  # ubuntu linux


# -------------------------

# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# SVM


def SVM(path):
    _dcl = cl.DataCLean()
    final_df, df = _dcl.extract(path)
    # corpus = model.text_concat(final_df)
    li_clean_text = _dcl.clean_data(final_df)
    uniqueWords = _dcl.make_unique_li(li_clean_text)
    # print(uniqueWords)
    docVector = _dcl.DocVector(
        final_df, uniqueWords
    )  ###_dcl.DocVector or _dcl.binary_docvectir
    ###########################
    df = docVector.values
    X_train, Y = df[:, :-1], df[:, -1]
    Y_train = convert_to_0_or_1(Y)
    return (X_train, Y_train)


# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------


def convert_to_0_or_1(Y):
    Y_train = []
    for y in Y:
        if y == "yes":
            Y_train.append(1)
        else:
            Y_train.append(0)
    return Y_train


# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# Training


X, Y = SVM(path)

a = np.size(X, 0)
X_split = int(np.size(X, 0) * 0.7)


X_train = X[0:X_split]
Y_train = Y[0:X_split]

X_test = X[X_split:, :]
Y_test = Y[X_split:]


# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

class_weight = {0: 2, 1: 1}

# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# Applying SVM

# print(Y_train)
clf = svm.SVC(probability=True, C=1.0, kernel="linear", degree=3, gamma="auto")
model = clf.fit(X_train, Y_train)
print(model.score(X_test, Y_test))

# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# Applying Naive Bayes

Naive = naive_bayes.MultinomialNB()
Naive.fit(X_train, Y_train)
print(Naive.score(X_test, Y_test))

# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# statitics


def report_results(model, X, y):
    pred_proba = model.predict_proba(X)[:, 1]
    pred = model.predict(X)

    auc = roc_auc_score(y, pred_proba)
    acc = accuracy_score(y, pred)
    f1 = f1_score(y, pred)
    prec = precision_score(y, pred)
    rec = recall_score(y, pred)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    TrueNeg = tn / (tn + fp)
    result = {
        "auc": auc,
        "f1": f1,
        "acc": acc,
        "precision": prec,
        "recall": rec,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "True Negative rate": TrueNeg,
    }
    return result


# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# statitics for SVM

stats = report_results(model, X_test, Y_test)
print(stats)
