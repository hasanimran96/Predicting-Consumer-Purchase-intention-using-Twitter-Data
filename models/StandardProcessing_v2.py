# imports
import Clean as cl
import docVector as dv
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from textblob import TextBlob
from nltk.stem import PorterStemmer
from textblob import Word
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import naive_bayes
from sklearn import linear_model
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import make_scorer, accuracy_score, f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    recall_score,
    precision_score,
)
import os

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


def generatingTrainSet():
    _dcl = cl.DataCLean()
    final_df, uniqueWords = _dcl.Clean()
    _dv = dv.DocumentVector()
    # docVector = _dv.tf_idf(final_df, uniqueWords)
    docVector = _dv.DocVector(final_df, uniqueWords)
    # docVector = _dv.binary_docvector(final_df, uniqueWords)

    # -------------------------------------------------------------------------
    # using textblob dict approach
    # import NaiveBayesTextBlob as tb

    # polarity_docVector = tb.text_blob(docVector, uniqueWords)
    # docVector = polarity_docVector
    # -------------------------------------------------------------------------

    df = docVector.values
    X_train, Y = df[:, :-1], df[:, -1]
    Y_train = convert_to_0_or_1(Y)
    return (X_train, Y_train)
# -------------------------------------------------------------------------

# -------------------------------------------------------------------------


def extract(path):
    fd = open(path, encoding="utf-8", errors="replace")
    df = pd.read_csv(fd)
    defined = df["class"] != ("undefined")
    # #output dataframe without undeined
    df2 = df[defined]
    defined1 = df2["class"] != "Undefined"
    df4 = df2[defined1]
    # replace no PI with no
    df3 = df4.replace("No PI", "no")
    # replace PI with yes
    final = df3.replace("PI", "yes")
    replace_yes = final.replace("Yes", "yes")
    final_df = replace_yes.replace("No", "no")
    return final_df, df
# -------------------------------------------------------------------------

# -------------------------------------------------------------------------


def report_results(model, X, y):
    # print(X.shape)
    # print(y.shape)
    pred_proba = model.predict_proba(X)[:, 1]
    pred = model.predict(X)
    # print(type(pred_proba))
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
    return result, pred
# -------------------------------------------------------------------------


def output_to_results(model):
    # -------------------------------------------------------------------------
    # SPLITTING THE DATA
    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

    X, Y = generatingTrainSet()

    a = np.size(X, 0)
    X_split = int(np.size(X, 0) * 0.7)

    X_train = X[0:X_split]
    Y_train = Y[0:X_split]

    X_test = X[X_split:, :]
    Y_test = Y[X_split:]

    class_weight = {0: 2, 1: 1}
   # -------------------------------------------------------------------------

   # -----------------------------------------------------------------------
   # Select model to train and display stats
   # -----------------------------------------------------------------------
    if model == "SVM":
        SVM = svm.SVC(probability=True, C=1.0,
                      kernel="linear", degree=3, gamma="auto")
        SVM.fit(X_train, Y_train)
        stats, pred = report_results(SVM, X_test, Y_test)
    elif model == "Naive Bayes":
        Naive = naive_bayes.MultinomialNB()
        Naive.fit(X_train, Y_train)
        stats, pred = report_results(Naive, X_test, Y_test)
    elif model == "Logistic Regression":
        logisticReg = linear_model.LogisticRegression(C=1.0)
        logisticReg.fit(X_train, Y_train)
        stats, pred = report_results(logisticReg, X_test, Y_test)
    elif model == "Decision Tree":
        DTC = DecisionTreeClassifier(min_samples_split=7, random_state=252)
        DTC.fit(X_train, Y_train)
        stats, pred = report_results(DTC, X_test, Y_test)
    elif model == "Neural Network":
        neural_network = MLPClassifier(
            solver="lbfgs", alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1
        )
        neural_network.fit(X_train, Y_train)
        stats, pred = report_results(neural_network, X_test, Y_test)
        # -----------------------------------------------------------------------
    return stats


def read_dir():

    path = "uploadeddata\\"
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        files.append(f)
    return files


print(output_to_results("SVM"))
# output_to_results("uploadeddata\AnnotatedData2.csv", "TF-IDF", "Naive Bayes")
# output_to_test("uploadeddata\Annotated4.csv","uploadeddata\AnnotatedData2.csv", "TF-IDF", "Naive Bayes")
